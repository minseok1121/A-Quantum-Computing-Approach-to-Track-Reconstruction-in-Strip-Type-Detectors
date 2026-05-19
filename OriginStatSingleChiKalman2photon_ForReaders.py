import sys
from unittest.mock import MagicMock

# threadpoolctl이 사고 치기 전에 가짜(Mock)로 대체해서 에러 발생 차단
sys.modules['threadpoolctl'] = MagicMock()

import os
import gc
import glob
import uproot
import numpy as np
import pandas as pd
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg') 
# --- [설정값: 모든 프로세스가 공유] ---
#STRIP_Z_LIST = [1.0, 1.7, 2.4, 4.4, 6.4, 8.4]
STRIP_Z_LIST = [2.3, 4.3, 6.3, 8.3, 10.3]
STRIP_POS = np.linspace(-5.0 + (10.0/128)/2, 5.0 - (10.0/128)/2, 128)
PAD_SIZE = 1.0
OUTPUT_DIR = "KalmanHybrid_Visualize_Batch_Analysis_Results_DoublePhoton"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib.cm as cm # 구버전 호환용

# --- [기존 유틸리티 함수] ---
def to_stereo(x, y):
    inv_sqrt2 = 0.70710678
    return (x - y) * inv_sqrt2, (x + y) * inv_sqrt2

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def apply_kalman_tracking_two_photons(ransac_pool, csi_pool, num_iterations=30, output_dir="tracking_plots_two_photon"):
    """
    K-Means를 사용하여 두 개의 포톤 클러스터를 분리하고, 
    각 클러스터에 대해 독립적으로 칼만 트래킹을 수행합니다.
    """
    pts = np.array(ransac_pool)
    csi_data = np.array(csi_pool)
    
    # 두 포톤을 분리하려면 최소한의 데이터 포인트가 필요합니다.
    if len(pts) < 6: 
        print("Error: Not enough points for two-photon tracking.")
        return None
    
    # 저장 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # [STEP 1] XY 정사영 기반 K-Means 클러스터링 (2개로 분리)
    # RWELL 데이터를 XY 평면에 투영하여 두 덩어리로 나눕니다.
    xy_coords = pts[:, 1:3]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(xy_coords)
    labels = kmeans.labels_

    # 전체 결과를 담을 리스트
    all_tracks_pts = []
    all_tracks_slopes = []

    # [STEP 2] 각 클러스터(포톤)별 독립 루프
    for cluster_idx in range(2):
        # 해당 클러스터의 RWELL 포인트만 추출
        c_pts = pts[labels == cluster_idx]
        if len(c_pts) < 3: continue
        px, py = np.mean(c_pts[:, 1]), np.mean(c_pts[:, 2])
        # CsI 데이터 결합 (0.1 이상의 에너지만)
        combined_obs = [[p[0], p[1], p[2], 1.0, 0] for p in c_pts] + \
                       [[c[0], c[1], c[2], c[3], 1] for c in csi_data]
        combined_obs = np.array(combined_obs)
        
        rwell_zs = np.sort(np.unique(c_pts[:, 0]))
        unique_zs = np.sort(np.unique(combined_obs[:, 0]))
        if len(rwell_zs) >= 3:
            unique_zs = unique_zs[unique_zs <= rwell_zs[2]]

        obs_list = [combined_obs[combined_obs[:, 0] == cz] for cz in unique_zs]
        
        # 2. 칼만 트래킹 이터레이션
        # 초기 상태: [x, y, dx/dz, dy/dz]
        state = np.array([px, py, 0.0, 0.0])
        z0 = unique_zs[0]

        for it in range(num_iterations):
            current_z = z0
            Q = np.diag([1e-8, 1e-8, 1e-7, 1e-7]) 
            P = np.diag([0.1, 0.1, 1.0, 1.0])
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            temp_results = []

            for i in range(len(obs_list)):
                target_z = unique_zs[i]
                dz = target_z - current_z
                F = np.array([[1, 0, dz, 0], [0, 1, 0, dz], [0, 0, 1, 0], [0, 0, 0, 1]])
                
                # 1. Predict
                state = F @ state
                P = F @ P @ F.T + Q
                pred_pos = H @ state
                
                layer_data = obs_list[i]
                is_csi = (layer_data[0, 4] == 1)
                
                # 가중 평균 측정값 계산 (v_raw)
                dists = np.linalg.norm(layer_data[:, 1:3] - pred_pos, axis=1)
                window = 0.5 if not is_csi else 1.5
                weights = np.exp(-dists**2 / (2 * window**2)) * layer_data[:, 3]
                
                if np.sum(weights) > 1e-5:
                    v_x_raw = np.sum(layer_data[:, 1] * weights) / np.sum(weights)
                    v_y_raw = np.sum(layer_data[:, 2] * weights) / np.sum(weights)
                    # 가중치 합에 따른 동적 R 설정 가능 (여기선 일단 고정)
                    r_val = 0.05**2 if not is_csi else 0.5**2
                    z_k = np.array([v_x_raw, v_y_raw]) # 순수 측정값 사용
                    
                    # 2. Update (칼만 필터 공식에 충실)
                    S = H @ P @ H.T + (np.eye(2) * r_val)
                    K = P @ H.T @ np.linalg.inv(S)
                    state = state + K @ (z_k - pred_pos)
                    P = (np.eye(4) - K @ H) @ P
                else:
                    # 측정값이 없으면 예측치만 유지하고 P만 키움 (Q 반영)
                    pass

                temp_results.append({'z': target_z, 'x': state[0], 'y': state[1], 'type': layer_data[0, 4]})
                current_z = target_z

            # 시각화 (선택 사항: 매 5회 및 마지막 이터레이션)
            if it == num_iterations - 1:
                plt.figure(figsize=(8, 8))
                plt.scatter(c_pts[:, 1], c_pts[:, 2], c='blue', alpha=0.3, label='Cluster Hits')
                v_hits = np.array([[r['x'], r['y']] for r in temp_results])
                plt.plot(v_hits[:, 0], v_hits[:, 1], 'r-o', label='Track Path')
                plt.title(f"Photon {cluster_idx} - Slope: {state[2]:.4f}, {state[3]:.4f}")
                plt.xlabel("X [cm]"); plt.ylabel("Y [cm]")
                plt.legend(); plt.grid(True)
                plt.savefig(f"{output_dir}/photon_{cluster_idx}_track.png")
                plt.close()

        # 결과 저장
        rwell_only_pts = np.array([[r['z'], r['x'], r['y'], cluster_idx] 
                                   for r in temp_results if r['type'] == 0])
        
        all_tracks_pts.append(rwell_only_pts)
        all_tracks_slopes.append((state[2], state[3]))

    # 모든 트랙의 점들을 하나의 배열로 합칩니다.
    # 분석 코드에서 kf_pts_labeled[:, 3] == r_idx 로 접근하기 위함입니다.
    if len(all_tracks_pts) > 0:
        combined_pts_labeled = np.vstack(all_tracks_pts)
    else:
        combined_pts_labeled = np.empty((0, 4))

    return combined_pts_labeled, all_tracks_slopes
