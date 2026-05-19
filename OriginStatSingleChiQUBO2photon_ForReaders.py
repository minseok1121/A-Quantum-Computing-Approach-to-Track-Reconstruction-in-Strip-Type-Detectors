import sys
from unittest.mock import MagicMock

# threadpoolctl이 사고 치기 전에 가짜(Mock)로 대체해서 에러 발생 차단
sys.modules['threadpoolctl'] = MagicMock()

import os
os.environ["THREADPOOLCTL_SKIP_MODULE_CHECK"] = "True"
# ... 나머지 환경변수 설정
# 1. OpenBLAS 및 시스템 쓰레드 강제 제한 (가장 중요)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
import matplotlib.transforms as mtransforms

import matplotlib
matplotlib.use('Agg') 
from dwave.system import DWaveSampler, EmbeddingComposite
# --- [설정값: 모든 프로세스가 공유] ---
#STRIP_Z_LIST = [1.0, 1.7, 2.4, 4.4, 6.4, 8.4]
STRIP_Z_LIST = [2.3, 4.3, 6.3, 8.3, 10.3]
STRIP_POS = np.linspace(-5.0 + (10.0/128)/2, 5.0 - (10.0/128)/2, 128)
PAD_SIZE = 1.0
OUTPUT_DIR = "QUBO_VisualizeV4_DoublePhoton"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib.cm as cm # 구버전 호환용

# --- [기존 유틸리티 함수] ---
def to_stereo(x, y):
    inv_sqrt2 = 0.70710678
    return (x - y) * inv_sqrt2, (x + y) * inv_sqrt2

import dimod
import minorminer
import random
import numpy as np
from scipy.stats import linregress

import dwave.inspector  # 시각화 툴 추가q
from dwave.system import DWaveSampler, EmbeddingComposite
try:
    _SAMPLER = EmbeddingComposite(DWaveSampler())
except Exception as e:
    print(f"Sampler Init Error: {e}")
    _SAMPLER = None


def solve_layer_zoom_qubo_log(layers_candidates):
    Q = {}
    n_layers = len(layers_candidates)
    
    # 1. 각 레이어의 노드 시작 인덱스(offset) 계산 (동적 ID 생성)
    node_offsets = [0]
    for i in range(n_layers):
        node_offsets.append(node_offsets[-1] + len(layers_candidates[i]))
    
    def get_id(l, s): 
        return node_offsets[l] + s

    # 하이퍼파라미터
    w_score = 0.0    # 데이터가 있는 곳을 선택하려는 힘 (매우 강하게)
    w_dist = 0.0       # 인접 레이어 간 연결성
    w_bend = 10.0      # 직선성 (앞 두 레이어가 세 번째를 결정하도록)
    penalty = 1000.0   # 레이어당 하나 선택 제약 (매우 강하게)

    def get_layer_importance(l):
        # 레이어 순서에 따른 가중치 (앞쪽 레이어 중시)
        return 1.0 ** (1 - l)

    # 1. Unary Terms: 각 노드의 개별 점수 (Score)
    for l in range(n_layers):
        l_imp = get_layer_importance(l)
        for s in range(len(layers_candidates[l])):
            idx = get_id(l, s)
            score = layers_candidates[l][s]['score']
            # 높은 score를 선택하도록 -log(score) 최소화
            Q[(idx, idx)] = -np.log(score * w_score * l_imp + 1e-6)

    # 2. Binary Terms: 연결 거리 및 3점 직선성
    for l in range(n_layers - 1):
        l_imp_trans = get_layer_importance(l)
        n_sub1 = len(layers_candidates[l])
        n_sub2 = len(layers_candidates[l+1])
        
        for s1 in range(n_sub1):
            for s2 in range(n_sub2):
                idx_a, idx_b = get_id(l, s1), get_id(l+1, s2)
                p1 = np.array(layers_candidates[l][s1]['center'])
                p2 = np.array(layers_candidates[l+1][s2]['center'])
                
                # 두 점 사이의 거리 비용
                dist_sq = np.sum((p2 - p1)**2)
                log_trans = np.log(100*dist_sq * l_imp_trans + 1.0) * w_dist
                Q[(idx_a, idx_b)] = Q.get((idx_a, idx_b), 0) + log_trans

                # 3점 직선성 (Bend Error) - 최소 3개 레이어가 있을 때만 작동
                if l < n_layers - 2:
                    l_imp_bend = get_layer_importance(l) * 10.0
                    n_sub3 = len(layers_candidates[l+2])
                    for s3 in range(n_sub3):
                        idx_c = get_id(l+2, s3)
                        p3 = np.array(layers_candidates[l+2][s3]['center'])
                        
                        # 2*p2 - p1 - p3 가 0에 가까울수록 직선
                        bend_error_sq = np.sum((2 * p2 - p1 - p3)**2)
                        #cost_bend = np.log(bend_error_sq * l_imp_bend + 1.0) * w_bend
                        cost_bend = bend_error_sq * l_imp_bend * w_bend
                        
                        # 삼항 관계를 이항 결합으로 분해 투영
                        Q[(idx_a, idx_b)] = Q.get((idx_a, idx_b), 0) + cost_bend
                        Q[(idx_b, idx_c)] = Q.get((idx_b, idx_c), 0) + cost_bend
                        Q[(idx_a, idx_c)] = Q.get((idx_a, idx_c), 0) + cost_bend

    # 3. Constraint: 레이어당 무조건 '하나'만 선택 (One-Hot Constraint)
    for l in range(n_layers):
        n_sub = len(layers_candidates[l])
        for s1 in range(n_sub):
            for s2 in range(s1 + 1, n_sub):
                idx1, idx2 = get_id(l, s1), get_id(l, s2)
                Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) + penalty

    # 샘플링 실행
    try:
        sampleset = _SAMPLER.sample_qubo(Q, num_reads=100)
        sample = sampleset.first.sample
    except Exception as e:
        print(f"QUBO Sampling Failed: {e}")
        return [layer[0] for layer in layers_candidates if len(layer) > 0]

    # 결과 추출
    final_path = []
    for l in range(n_layers):
        selected = [s for s in range(len(layers_candidates[l])) if sample.get(get_id(l, s), 0) == 1]
        if selected:
            idx = selected[0]
        else:
            # 선택된 노드가 없으면 가장 높은 score 노드로 대체
            idx = np.argmax([c['score'] for c in layers_candidates[l]]) if len(layers_candidates[l]) > 0 else 0
        
        if len(layers_candidates[l]) > 0:
            final_path.append(layers_candidates[l][idx])
            
    return final_path

def run_single_photon_tracking(cluster_pts, csi_pool, pid, file_label, iterations=10):
    csi_pool = np.asanyarray(csi_pool)
    cluster_pts = np.asanyarray(cluster_pts)
    
    unique_zs = np.sort(np.unique(cluster_pts[:, 0]))
    
    # [수정] 데이터가 존재하는 레이어 중 앞에서부터 '최대 3개'만 사용
    target_zs = unique_zs[:3] 
    if len(target_zs) == 0: return None

    # --- 초기 관심 영역(ROI) 설정 ---
    current_rois = []
    for z in target_zs:
        layer_hits = cluster_pts[cluster_pts[:, 0] == z]
        is_stereo = z in [4.3, 8.3]
        
        if len(layer_hits) > 0:
            if is_stereo:
                # 45도 회전된 좌표계에서 범위를 계산
                rot_x, rot_y = to_stereo(layer_hits[:, 1], layer_hits[:, 2])
                width_x = np.max(rot_x) - np.min(rot_x)
                width_y = np.max(rot_y) - np.min(rot_y)
                l_cx_rot, l_cy_rot = np.mean(rot_x), np.mean(rot_y)
                
                l_cx = (l_cx_rot + l_cy_rot) / (2 * 0.70710678)
                l_cy = (l_cy_rot - l_cx_rot) / (2 * 0.70710678)
            else:
                width_x = np.max(layer_hits[:, 1]) - np.min(layer_hits[:, 1])
                width_y = np.max(layer_hits[:, 2]) - np.min(layer_hits[:, 2])
                l_cx, l_cy = np.mean(layer_hits[:, 1]), np.mean(layer_hits[:, 2])
            
            raw_range = max(width_x, width_y)
            dynamic_size = raw_range * 1.0 + 0.0 * len(layer_hits)
        else:
            dynamic_size = 2.0
            l_cx, l_cy = np.mean(cluster_pts[:, 1]), np.mean(cluster_pts[:, 2])

        current_rois.append({'center': (l_cx, l_cy), 'size': dynamic_size})

    # --- 메인 반복문 (Iterations) ---
    for it in range(iterations):
        layers_candidates = []
        fig, axes = plt.subplots(1, len(target_zs), figsize=(5*len(target_zs), 5))
        if len(target_zs) == 1: axes = [axes]

        for i, z in enumerate(target_zs):
            roi = current_rois[i]
            rcx, rcy = roi['center']
            s = roi['size']
            is_stereo = z in [4.3, 8.3]
            
            layer_hits_all = csi_pool[csi_pool[:, 0] == z]
            layer_cluster_hits = cluster_pts[cluster_pts[:, 0] == z]
            next_s = s * 0.75 if it < iterations - 1 else s
            
            # 1. 마스크 필터링 (Stereo 여부에 따라 축 변경)
            if is_stereo:
                curr_rot_x, curr_rot_y = to_stereo(layer_cluster_hits[:, 1], layer_cluster_hits[:, 2])
                rcx_rot, rcy_rot = to_stereo(rcx, rcy)
                in_roi_mask = (np.abs(curr_rot_x - rcx_rot) < s/2) & (np.abs(curr_rot_y - rcy_rot) < s/2)
            else:
                in_roi_mask = (np.abs(layer_cluster_hits[:, 1] - rcx) < s/2) & (np.abs(layer_cluster_hits[:, 2] - rcy) < s/2)
            
            base_cx, base_cy = rcx, rcy
            if np.any(in_roi_mask):
                hits_in_roi = layer_cluster_hits[in_roi_mask]
                # 줌인 시 잘릴 위험 체크 (회전축 기준)
                if is_stereo:
                    h_rot_x, h_rot_y = to_stereo(hits_in_roi[:, 1], hits_in_roi[:, 2])
                    out_of_next_zoom = (np.abs(h_rot_x - rcx_rot) > next_s/2) | (np.abs(h_rot_y - rcy_rot) > next_s/2)
                else:
                    out_of_next_zoom = (np.abs(hits_in_roi[:, 1] - rcx) > next_s/2) | (np.abs(hits_in_roi[:, 2] - rcy) > next_s/2)
                
                shift_threshold = 0.4 if len(hits_in_roi) < 3 else 0.8
                if np.mean(out_of_next_zoom) > shift_threshold:
                    base_cx, base_cy = np.mean(hits_in_roi[:, 1]), np.mean(hits_in_roi[:, 2])

            # 2. 후보지 생성 (원래 좌표계에서 shift 생성)
            shift = next_s * 0.8
            if is_stereo:
                candidate_centers = [(base_cx, base_cy), (base_cx + shift * 0.7071, base_cy + shift * 0.7071), (base_cx - shift * 0.7071, base_cy - shift * 0.7071), (base_cx - shift * 0.7071, base_cy + shift * 0.7071), (base_cx + shift * 0.7071, base_cy - shift * 0.7071)]
            else:
                candidate_centers = [(base_cx, base_cy), (base_cx+shift, base_cy), (base_cx-shift, base_cy), (base_cx, base_cy+shift), (base_cx, base_cy-shift)]
            
            sub_nodes = []
            total_layer_energy = np.sum(layer_hits_all[:, 3]) if len(layer_hits_all) > 0 else 1.0
            n_total_hits = len(layer_cluster_hits)

            for n_cx, n_cy in candidate_centers:
                # 에너지 및 히트 카운트 마스크 (Stereo 대응)
                if is_stereo:
                    all_rot_x, all_rot_y = to_stereo(layer_hits_all[:, 1], layer_hits_all[:, 2])
                    n_rot_cx, n_rot_cy = to_stereo(n_cx, n_cy)
                    energy_mask = (np.abs(all_rot_x - n_rot_cx) < next_s/2) & (np.abs(all_rot_y - n_rot_cy) < next_s/2)
                    
                    cls_rot_x, cls_rot_y = to_stereo(layer_cluster_hits[:, 1], layer_cluster_hits[:, 2])
                    cluster_mask = (np.abs(cls_rot_x - n_rot_cx) < next_s/2) & (np.abs(cls_rot_y - n_rot_cy) < next_s/2)
                else:
                    energy_mask = (np.abs(layer_hits_all[:, 1] - n_cx) < next_s/2) & (np.abs(layer_hits_all[:, 2] - n_cy) < next_s/2)
                    cluster_mask = (np.abs(layer_cluster_hits[:, 1] - n_cx) < next_s/2) & (np.abs(layer_cluster_hits[:, 2] - n_cy) < next_s/2)
                
                area_energy = np.sum(layer_hits_all[energy_mask, 3]) if np.any(energy_mask) else 0
                n_hits_in_area = np.sum(cluster_mask)
                
                if n_hits_in_area > 0:
                    score = (area_energy/total_layer_energy + n_hits_in_area/n_total_hits) / np.log1p(n_total_hits) + 0.001
                    sub_nodes.append({'center': (n_cx, n_cy), 'size': next_s, 'score': score})

            if not sub_nodes: sub_nodes.append({'center': (base_cx, base_cy), 'size': next_s, 'score': 0.0001})
            layers_candidates.append(sub_nodes)

            # 3. 시각화 (Stereo는 빨간색 45도 회전 사각형)
            ax = axes[i]
            ax.scatter(layer_cluster_hits[:, 1], layer_cluster_hits[:, 2], color='blue', alpha=0.3)
            for node in sub_nodes:
                if is_stereo:
                    rect = plt.Rectangle((node['center'][0] - node['size']/2, node['center'][1] - node['size']/2), 
                                        node['size'], node['size'], fill=False, color='red', alpha=0.5)
                    trans = mtransforms.Affine2D().rotate_around(node['center'][0], node['center'][1], np.deg2rad(45)) + ax.transData
                    rect.set_transform(trans)
                    ax.add_patch(rect)
                else:
                    ax.add_patch(plt.Rectangle((node['center'][0]-node['size']/2, node['center'][1]-node['size']/2), 
                                                node['size'], node['size'], fill=False, color='gray', alpha=0.5))
            ax.scatter(rcx, rcy, marker='x', color='black', s=50)
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.grid(True, linestyle='--', alpha=0.6)  # 점선 그리드
            ax.set_axisbelow(True)

        plt.tight_layout()
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        # 2. [핵심] 폴더가 없으면 생성 (exist_ok=True는 이미 폴더가 있어도 에러를 내지 않음)
        os.makedirs(debug_dir, exist_ok=True)
        # 3. 저장 실행
        save_path = os.path.join(debug_dir, f"{file_label}_PID{int(pid)}_it{it:02d}.png")
        plt.savefig(save_path)
        plt.close(fig)

        # QUBO 최적화로 다음 ROI 중심 결정
        try:
            current_rois = solve_layer_zoom_qubo_log(layers_candidates)
        except Exception as e:
            print(f"QUBO Error: {e}"); break

    # 최종 결과 반환 및 기울기 계산
    final_pts = np.array([[z, current_rois[i]['center'][0], current_rois[i]['center'][1], pid] for i, z in enumerate(target_zs)])
    vx, _ = np.polyfit(final_pts[:, 0], final_pts[:, 1], 1)
    vy, _ = np.polyfit(final_pts[:, 0], final_pts[:, 2], 1)
    
    return final_pts, (vx, vy)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def apply_two_photon_split_reco(ransac_pool, csi_pool, file_label):
    pts = np.array(ransac_pool)
    if len(pts) < 6: return None  # 두 트랙을 찾기 위한 최소 힛 부족
    
    # --- [STEP 1] 2D 정사영 및 클러스터링 ---
    # 모든 레이어의 힛을 XY 평면에 투영하여 두 그룹으로 나눕니다.
    xy_coords = pts[:, 1:3]
    # n_clusters=2: 두 개의 포톤 궤적을 찾음
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(xy_coords)
    labels = kmeans.labels_
    # n_components는 클러스터 개수(K)와 같습니다.
    #gmm = GaussianMixture(n_components=2, random_state=42).fit(xy_coords)
    #labels = gmm.predict(xy_coords)

    all_tracks_pts = []
    all_slopes = []

    # --- [STEP 2] 각 클러스터(포톤)에 대해 독립적으로 Reco 실행 ---
    for pid in range(2):
        # 해당 클러스터에 할당된 힛들만 필터링
        cluster_hits = pts[labels == pid]
        
        # 유효한 레이어 수가 최소 3개는 되어야 추적 가능
        unique_zs = np.sort(np.unique(cluster_hits[:, 0]))
        if len(unique_zs) < 3:
            continue
            
        # 기존에 정의된 싱글 포톤 추적 로직 호출
        # 결과값: (pts_with_pid, slope_tuple)
        result = run_single_photon_tracking(cluster_hits, csi_pool, pid, file_label)
        
        if result is not None:
            track_pts, slope = result
            all_tracks_pts.append(track_pts)
            all_slopes.append(slope)

    if not all_tracks_pts: 
        return None

    # --- [STEP 3] 검증 로직 형식에 맞게 통합 ---
    # kf_pts_labeled: [z, x, y, pid] 형태의 모든 점 병합
    kf_pts_labeled = np.vstack(all_tracks_pts)
    # slopes: [(vx0, vy0), (vx1, vy1)]
    return kf_pts_labeled, all_slopes
