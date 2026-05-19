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

def plot_event_layers(csi_pool, ransac_pool, t_infos, event_id="Unknown"):
    """
    csi_pool: [[z, x, y, energy, type], ...]
    ransac_pool: [[z, x, y, 0, 0], ...]
    t_infos: MC Truth 정보 (pos, vec 포함)
    """
    # 2x3 레이아웃 (5개 레이어 + 1개 빈칸)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle(f"Event Display: CsI & RWELL with MC Truth (Event: {event_id})", fontsize=16)

    axes_flat = axes.flatten()

    if len(csi_pool) == 0 and len(ransac_pool) == 0:
        plt.close()
        return
    # 컬러맵 설정 (구버전 호환 및 복사본 사용)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps['YlOrRd'].copy()
    except:
        import copy
        cmap = copy.copy(cm.get_cmap('YlOrRd'))
    cmap.set_bad(color='white')
    
    csi_np = np.array(csi_pool) if len(csi_pool) > 0 else np.array([])
    rwell_np = np.array(ransac_pool) if len(ransac_pool) > 0 else np.array([])
    
    max_energy = csi_np[:, 3].max() if len(csi_np) > 0 else 1.0

    for i in range(6):
        ax = axes_flat[i]
        if i < 5:
            # [최적화 2] 유동적 rwell_zs[i] 대신 고정된 물리 좌표 사용
            curr_z = STRIP_Z_LIST[i]
            
            # --- [Part 1: CsI 에너지 플롯] ---
            if csi_np.size > 0:
                # 물리적 RWELL Z보다 살짝 앞(0 < dz < 1.5)에 있는 CsI 데이터 필터링
                diff = csi_np[:, 0] - curr_z
                mask = (diff > 0) & (diff < 1.5)
                layer_csi = csi_np[mask]
                
                if len(layer_csi) > 0:
                    rects = [patches.Rectangle((row[1]-0.5, row[2]-0.5), 1.0, 1.0) for row in layer_csi]
                    pc = PatchCollection(rects, cmap=cmap, alpha=0.7, edgecolors='gray', linewidths=0.5)
                    pc.set_array(layer_csi[:, 3])
                    pc.set_clim(0, max_energy)
                    ax.add_collection(pc)

            # --- [Part 2: RWELL 히트 플롯] ---
            if rwell_np.size > 0:
                # 고정된 curr_z와 일치하는 히트만 플롯
                mask = np.isclose(rwell_np[:, 0].astype(float), float(curr_z))
                layer_rwell = rwell_np[mask]
                if len(layer_rwell) > 0:
                    ax.scatter(layer_rwell[:, 1], layer_rwell[:, 2], 
                               s=60, c='blue', marker='x', linewidths=2, label='RWELL Hit')

            # --- [Part 3: MC Truth] ---
            for t_idx, info in enumerate(t_infos):
                p0, vec = info['pos'], info['vec']
                if vec[2] != 0:
                    scale = (curr_z - p0[2]) / vec[2]
                    tx, ty = p0[0] + vec[0] * scale, p0[1] + vec[1] * scale
                    ax.scatter(tx, ty, s=200, edgecolors='black', facecolors='none', marker='*', linewidths=1.5)

            # [핵심] 이제 curr_z는 항상 숫자이므로 포맷팅 에러가 절대 안 남
            ax.set_title(f"Layer {i+1} (z={curr_z:.1f}cm)")
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
        else:
            ax.axis('off')

        # 공통 축 설정
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        if i >= 3: ax.set_xlabel("X [cm]")
        if i % 3 == 0: ax.set_ylabel("Y [cm]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/event_{event_id.replace('.root', '')}_fixed.png")
    plt.close()

# --- [기존 유틸리티 함수] ---
def to_stereo(x, y):
    inv_sqrt2 = 0.70710678
    return (x - y) * inv_sqrt2, (x + y) * inv_sqrt2

def plot_debug_event(evt_id, ransac_pool, kf_pts_labeled, t_infos, errors):
    """kf_pts_labeled: [z, x, y, pid] 형태의 넘파이 배열"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 컬러맵 설정 (Photon 0: Red, Photon 1: Green)
    reco_colors = ['red', 'green']
    truth_colors = ['blue', 'cyan']

    for i, z_s in enumerate(STRIP_Z_LIST):
        ax = axes[i]
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        
        # 1. RANSAC Pool
        pool_layer = [p for p in ransac_pool if p[0] == z_s]
        if pool_layer:
            pool_layer = np.array(pool_layer)
            ax.scatter(pool_layer[:,1], pool_layer[:,2], color='gray', alpha=0.3, s=30, label='Candidates')

        # 2. MC Truth (모든 Truth 그리기)
        for t_idx, t_info in enumerate(t_infos):
            tx = t_info['x'] + (t_info['px']/t_info['pz']) * z_s
            ty = t_info['y'] + (t_info['py']/t_info['pz']) * z_s
            ax.scatter(tx, ty, color=truth_colors[t_idx % 2], facecolors='none', 
                       edgecolors=truth_colors[t_idx % 2], s=150, 
                       label=f'Truth {t_idx}' if i==0 else "")

        # 3. Reco Hits (PID에 따라 색상 구분)
        layer_reco = kf_pts_labeled[np.abs(kf_pts_labeled[:,0] - z_s) < 0.01]
        for row in layer_reco:
            pid = int(row[3])
            ax.scatter(row[1], row[2], color=reco_colors[pid % 2], marker='x', 
                       s=100, linewidths=2, label=f'Reco {pid}' if i==0 else "")

        ax.set_title(f"Layer {i+1}"); ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax.legend(loc='upper right', fontsize='x-small')

    err_str = "\n".join([f"Trk{k}: Dist={e[0]:.3f}, Ang={e[1]:.3f}, Mass={e[2]:.3f}" for k, e in errors.items()])
    plt.suptitle(f"Double Photon Debug [{evt_id}]\n{err_str}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Debug_{evt_id}.png")
    plt.close()


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

        # --- 기존 칼만 알고리즘 루틴 시작 ---
        """
        # 1. 전 레이어 정사영 ROI 필터링 (해당 클러스터 기준)
        projection_map = {}
        for c in csi_data:
            pos = (round(c[1], 1), round(c[2], 1)) 
            projection_map[pos] = projection_map.get(pos, 0) + c[3]

        max_energy = -1
        # 클러스터의 현재 평균 위치를 초기 추측값으로 사용
        px, py = np.mean(c_pts[:, 1]), np.mean(c_pts[:, 2])
        
        for (ux, uy) in projection_map.keys():
            current_2x2_sum = sum(projection_map.get((round(ux+dx, 1), round(uy+dy, 1)), 0) 
                                  for dx, dy in [(0,0), (1.0,0), (0,1.0), (1.0,1.0)])
            if current_2x2_sum > max_energy:
                # 현재 클러스터 중심에서 5cm 이내의 CsI 에너지만 유효한 것으로 판단 (포톤간 간섭 방지)
                if np.hypot(ux - px, uy - py) < 2.0:
                    max_energy = current_2x2_sum
                    px, py = ux + 0.5, uy + 0.5

        roi_mask = (np.abs(c_pts[:, 1] - px) <= 1.5) & (np.abs(c_pts[:, 2] - py) <= 1.5)
        filtered_pts = c_pts[roi_mask] if np.sum(roi_mask) >= 3 else c_pts
        """
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

def process_single_file(f_path):
    local_stats = []
    local_total = 0
    local_passed = 0
    
    try:
        with uproot.open(f_path) as file:
            local_total += 1
            tree = file["DAMSA"]
            
            # [최적화 1] 로드 단계에서 컷 (z=0 또는 전자/양전자만)
            raw_df = tree.arrays(
                ["PDGID", "x", "y", "z", "PPIPZ", "px", "py", "pz"],
                cut="(z == 0) | (abs(PDGID) == 11)",
                library="pd"
            )
            
            # [수정] 루프가 없으므로 continue 대신 return 사용
            if raw_df.empty: 
                return local_total, local_passed, local_stats

            raw_df[['x', 'y', 'z']] /= 10.0

# --- [최적화 2] MC Truth 추출 (2개 포톤 대응) ---
            t_mask = (np.abs(raw_df['z']) < 0.001) & (raw_df['PDGID'] == 22) & (raw_df['PPIPZ'] == 0)
            t_df = raw_df[t_mask].copy()

            # [추가] 각 포톤의 에너지(모멘텀 크기) 계산
            t_df['energy'] = np.sqrt(t_df['px']**2 + t_df['py']**2 + t_df['pz']**2)
            
            # [추가] 두 포톤 모두 에너지가 500 이상인 이벤트만 남기기
            # (이벤트 내의 모든 t_df 행이 500 이상이어야 함)
            if not (len(t_df) >= 1 and (t_df['energy'] >= 500).all()):
                return local_total, local_passed, local_stats

            # 두 포톤의 정보를 리스트에 담기
            t_infos = []
            for idx in range(len(t_df)):
                row = t_df.iloc[idx]
                t_vec = np.array([row['px'], row['py'], row['pz']])
                t_energy = row['energy']
                
                if t_energy > 0: t_vec /= t_energy # 정규화
                
                t_infos.append({
                    'pos': np.array([row['x'], row['y'], row['z']]),
                    'vec': t_vec,
                    'energy': t_energy, # 에너지 정보 저장
                    'px': row['px'], 'py': row['py'], 'pz': row['pz'],
                    'x': row['x'], 'y': row['y']
                })
            
            # [최적화 3] 필요한 컬럼만 슬라이싱해서 복사 (메모리 해제 유도)
            event_df = raw_df[np.abs(raw_df['PDGID']) == 11][['x', 'y', 'z', 'PPIPZ']].copy()
            
            # --- [Part 1: CsI Energy Deposit Processing (1cm^3 Cube)] ---
            CSI_Z_BOUNDS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)] # CsI 층의 z 범위 (예시)
            csi_pool = []

            for z_min, z_max in CSI_Z_BOUNDS:
                # 해당 층의 CsI 에너지 디포짓 필터링
                # 수정 후
                csi_layer = event_df[
                    (event_df['z'] >= z_min) & (event_df['z'] < z_max) & (event_df['PPIPZ'] > 0)
                ].copy()  # <--- 명시적 복사 추가
                
                if csi_layer.empty: continue

                # 1cm^3 격자화를 위해 좌표를 정수화 (Grid Indexing)
                # x, y 좌표를 10mm(1cm) 단위로 floor 처리
                csi_layer['gx'] = np.floor(csi_layer['x'])
                csi_layer['gy'] = np.floor(csi_layer['y'])
                
                # 격자별 에너지 합산
                grid_energy = csi_layer.groupby(['gx', 'gy'])['PPIPZ'].sum().reset_index()

                for _, row in grid_energy.iterrows():
                    # 각 큐브의 중심 좌표로 변환 (0.5를 더해 중심값 계산)
                    center_x = (row['gx']) + 0.5
                    center_y = (row['gy']) + 0.5
                    center_z = (z_min + z_max) / 2.0
                    
                    # [z, x, y, energy, type] 형태 (type 1은 CsI, 0은 RWELL로 구분 가능)
                    csi_pool.append([center_z, center_x, center_y, row['PPIPZ'], 1])

            # --- [Part 2: Micro-RWELL Strip Processing (기존 유지 및 확장)] ---
            ransac_pool = []
            for idx, z_s in enumerate(STRIP_Z_LIST):
                l_strip = event_df[(np.abs(event_df['z'] - z_s) < 0.05) & (event_df['PPIPZ'] == 0)]
                if l_strip.empty: continue
                if idx == 1 or idx == 3: # Stereo
                    u, v = to_stereo(l_strip['x'].values, l_strip['y'].values)
                    v_idx = np.abs(u[:, None] - STRIP_POS[None, :]).argmin(axis=1)
                    u_idx = np.abs(v[:, None] - STRIP_POS[None, :]).argmin(axis=1)
                else:
                    v_idx = np.abs(l_strip['x'].values[:, None] - STRIP_POS[None, :]).argmin(axis=1)
                    u_idx = np.abs(l_strip['y'].values[:, None] - STRIP_POS[None, :]).argmin(axis=1)

                true_pads = set(zip(np.floor(l_strip['x']/PAD_SIZE), np.floor(l_strip['y']/PAD_SIZE)))
                for gv, gu in itertools.product(np.unique(v_idx), np.unique(u_idx)):
                    if idx == 1 or idx == 3:
                        su, sv = STRIP_POS[gv], STRIP_POS[gu]
                        gx, gy = (su + sv) * 0.70710678, (sv - su) * 0.70710678
                    else:
                        gx, gy = STRIP_POS[gv], STRIP_POS[gu]
                    
                    if (np.floor(gx/PAD_SIZE), np.floor(gy/PAD_SIZE)) in true_pads:
                        ransac_pool.append([z_s, gx, gy, 0, 0])
            """
            # --- [Part 2 루프 이후: Strict Truth-Cell Validation] ---
            is_event_valid = True
            #rwell_zs = STRIP_Z_LIST  # 5개 레이어의 Z 좌표 리스트
            rwell_zs = STRIP_Z_LIST[1:]

            # 넘파이 변환 (속도 및 인덱싱용)
            csi_np = np.array(csi_pool)
            rwell_np = np.array(ransac_pool)

            for curr_z in rwell_zs:
                for t_info in t_infos:
                    p0 = t_info['pos']
                    vec = t_info['vec']
                    
                    # 1. 현재 레이어(Z)에서 Truth의 정밀 XY 좌표 계산
                    scale = (curr_z - p0[2]) / vec[2]
                    tx = p0[0] + vec[0] * scale
                    ty = p0[1] + vec[1] * scale

                    # 2. Truth가 찍힌 "정확한 셀"의 중심 좌표 계산 (1cm 격자 기준)
                    # gx, gy는 셀의 좌하단 좌표, +0.5는 중심 좌표
                    target_cx = np.floor(tx) + 0.5
                    target_cy = np.floor(ty) + 0.5

                    # --- [조건 A: CsI Strict Check] ---
                    # 해당 레이어(z)에서 정확히 target_cx, target_cy인 셀이 있는지 확인
                    # z 매칭은 RWELL z보다 0~1.5cm 앞에 있는 CsI 층 탐색
                    csi_mask = (csi_np[:, 0] < curr_z) & (curr_z - csi_np[:, 0] < 1.5)
                    layer_csi = csi_np[csi_mask]
                    
                    # 논리 연산: (중심X 일치) AND (중심Y 일치)
                    cell_fired = np.any((np.abs(layer_csi[:, 1] - target_cx) < 0.01) & 
                                        (np.abs(layer_csi[:, 2] - target_cy) < 0.01))
                    
                    if not cell_fired:
                        is_event_valid = False
                        break

                    # --- [조건 B: RWELL Hit Check] ---
                    # RWELL은 스트립이므로 '정확한 셀' 개념보다는 Truth 주변 1.0cm 이내 히트 존재 여부로 판단
                    layer_rwell = rwell_np[rwell_np[:, 0] == curr_z]
                    rwell_fired = False
                    if len(layer_rwell) > 0:
                        dist_rwell = np.sqrt((layer_rwell[:, 1] - tx)**2 + (layer_rwell[:, 2] - ty)**2)
                        if np.any(dist_rwell < 0.3):
                            rwell_fired = True
                    
                    if not rwell_fired:
                        is_event_valid = False
                        break
                
                if not is_event_valid: break

            # 최종 필터링
            if not is_event_valid:
                # 어느 한 레이어라도 Truth가 가리킨 셀에 에너지가 없으면 스킵
                return local_total, local_passed, local_stats
            """
            # --- [Part 3: 이 아래는 검증된 이벤트만 통과함] ---
            local_passed += 1

            plot_event_layers(csi_pool, ransac_pool, t_infos, event_id=f_path.split('/')[-1])
            results = apply_kalman_tracking_two_photons(ransac_pool, csi_pool)

            if results is not None:
                kf_pts_labeled, slopes = results  # kf_pts_labeled: [z, x, y, pid], slopes: [(sx0, sy0), (sx1, sy1)]
                track_errors = {}
                matches = []

                # 1. 1:1 매칭 (Cost Matrix 기반 최단 거리 매칭)
                match_candidates = []
                true_sz = STRIP_Z_LIST[0]
                
                # 모든 Reco(r_idx)와 모든 Truth(t_idx) 조합의 거리 계산
                # kf_pts_labeled에서 각 pid별로 첫 번째 레이어(z_min)의 점을 찾아 거리를 잽니다.
                for r_idx in range(len(slopes)):
                    reco_pts = kf_pts_labeled[kf_pts_labeled[:, 3] == r_idx]
                    if len(reco_pts) == 0: continue
                    
                    # 첫 번째 레이어의 점 (z가 가장 작은 점)
                    r_start = reco_pts[np.argmin(reco_pts[:, 0])][1:3] # (x, y)
                    
                    for t_idx, t_info in enumerate(t_infos):
                        tx = t_info['x'] + (t_info['px']/t_info['pz']) * true_sz
                        ty = t_info['y'] + (t_info['py']/t_info['pz']) * true_sz
                        d = np.sqrt((r_start[0]-tx)**2 + (r_start[1]-ty)**2)
                        match_candidates.append({'r_idx': r_idx, 't_idx': t_idx, 'dist': d})

                # 거리가 짧은 순으로 정렬하여 1:1 할당
                match_candidates.sort(key=lambda x: x['dist'])
                assigned_reco, assigned_truth = set(), set()
                
                for cand in match_candidates:
                    if cand['r_idx'] not in assigned_reco and cand['t_idx'] not in assigned_truth:
                        matches.append((cand['r_idx'], cand['t_idx']))
                        assigned_reco.add(cand['r_idx'])
                        assigned_truth.add(cand['t_idx'])

                reco_results = {} # 각 r_idx에 대한 벡터와 에너지 정보를 임시 저장

                for r_idx, t_idx in matches:
                    reco_pts_subset = kf_pts_labeled[kf_pts_labeled[:, 3] == r_idx]
                    rsx, rsy = slopes[r_idx]
                    rz0, rx0, ry0 = reco_pts_subset[0, 0], reco_pts_subset[0, 1], reco_pts_subset[0, 2]
                    
                    best_t = t_infos[t_idx]
                    
                    # (A) Z=0 지점 거리
                    tx_at_z0, ty_at_z0 = best_t['x'], best_t['y']
                    rx_at_z0 = rx0 + rsx * (0 - rz0)
                    ry_at_z0 = ry0 + rsy * (0 - rz0)
                    dist = np.sqrt((rx_at_z0 - tx_at_z0)**2 + (ry_at_z0 - ty_at_z0)**2)
                    
                    # (B) 방향 각도 차이
                    r_vec = np.array([rsx, rsy, 1.0])
                    r_vec /= np.linalg.norm(r_vec)
                    t_vec = best_t['vec'] 
                    angle = np.arccos(np.clip(np.dot(t_vec, r_vec), -1.0, 1.0))

                    # (C) Reduced Chi2
                    chi2_sum = 0
                    for pt in reco_pts_subset:
                        pz, px, py = pt[0], pt[1], pt[2]
                        exp_x = best_t['x'] + (best_t['px']/best_t['pz']) * (pz)
                        exp_y = best_t['y'] + (best_t['py']/best_t['pz']) * (pz)
                        chi2_sum += (px - exp_x)**2 + (py - exp_y)**2
                    
                    ndf = len(reco_pts_subset) * 2
                    reduced_chi2 = chi2_sum / ndf if ndf > 0 else 999
                    
                    # --- [Invariant Mass용 데이터 수집] ---
                    # Truth의 에너지(E)와 Reco의 방향벡터(r_vec) 저장
                    reco_results[r_idx] = {
                        'vec': r_vec,
                        'energy': best_t['energy'], # Truth에서 가져온 에너지
                        'dist': dist,
                        'angle': angle,
                        'chi2': reduced_chi2
                    }

                # --- [Invariant Mass 계산 (두 포톤이 매칭되었을 때)] ---
                inv_mass = -1.0
                if len(reco_results) >= 2:
                    keys = list(reco_results.keys())
                    # 첫 두 매칭된 트랙 사용 (보통 ALP는 2개 포톤)
                    r1, r2 = reco_results[keys[0]], reco_results[keys[1]]
                    cos_theta = np.clip(np.dot(r1['vec'], r2['vec']), -1.0, 1.0)
                    inv_mass = np.sqrt(2 * r1['energy'] * r2['energy'] * (1 - cos_theta))

                # 통계 저장 및 track_errors 업데이트
                for r_idx, info in reco_results.items():
                    # (D) track_errors에 inv_mass 추가 (디버그 플랏 인자용)
                    track_errors[r_idx] = (info['dist'], info['angle'], inv_mass)
                    
                    local_stats.append({
                        'dist_start': info['dist'], 
                        'angle_diff': info['angle'], 
                        'norm_chi2': info['chi2'],
                        'inv_mass': inv_mass # 4번째 통계 추가
                    })

                # 3. 디버그 조건 체크 (inv_mass 정보 포함된 track_errors 전달)
                if any(d > 0.1 or a > 0.1 for d, a, m in track_errors.values()):
                    evt_label = os.path.basename(f_path).split('.')[0]
                    plot_debug_event(evt_label, ransac_pool, kf_pts_labeled, t_infos, track_errors)


            print(f"✅ 처리 완료: {os.path.basename(f_path)}")
            
        return local_total, local_passed, local_stats

    except Exception as e:
        print(f"❌ 에러 ({os.path.basename(f_path)}): {e}")
        return 0, 0, []

import pickle
from concurrent.futures import ProcessPoolExecutor  # 병렬 처리를 위한 라이브러리

def main():
    file_list = glob.glob("/gv0/Users/mioh/DAMSA_FULL/ms/ALPGun/ALP100DoublePhotonOrigin/DAMSA_m100_r*.root")
    PICKLE_PATH = f"{OUTPUT_DIR}/analysis_results.pkl"
    
    # --- [병렬 처리 핵심부] ---
    max_workers = 5 # 사용할 CPU 코어 수
    results = []
    
    print(f">>> {max_workers}개의 CPU 코어를 사용하여 병렬 처리를 시작합니다. (총 파일: {len(file_list)})")
    
    # ProcessPoolExecutor를 사용하여 process_single_file을 병렬 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map은 파일 리스트를 하나씩 함수에 전달하고 결과를 리스트로 반환합니다.
        results = list(executor.map(process_single_file, file_list))

    # 병렬 처리 후 개별 프로세스에서 남은 메모리 정리 (메인 프로세스용)
    plt.close('all')
    gc.collect()

    # --- [결과 저장: 기존 로직 유지] ---
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 모든 결과가 피클로 저장되었습니다: {PICKLE_PATH}")

    # [Step 3] 통계 취합 및 플로팅
    # results가 None이거나 에러가 난 경우를 대비해 필터링
    valid_results = [r for r in results if r is not None]
    
    total_processed = sum(r[0] for r in valid_results)
    total_passed = sum(r[1] for r in valid_results)
    combined_stats = []
    for r in valid_results:
        combined_stats.extend(r[2])

    print(f"📊 분석 요약: 총 {total_processed} 중 {total_passed} 이벤트 선택됨")

    if combined_stats:
        s_df = pd.DataFrame(combined_stats)
        s_df.to_pickle(f"{OUTPUT_DIR}/statistics_df.pkl")
        
        # 1x4 플랏으로 변경 (Inv Mass 추가)
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        # 1. Vertex Resolution
        axes[0].hist(s_df['dist_start'], bins=50, color='salmon', edgecolor='black')
        axes[0].set_title("Vertex Resolution (cm)"); axes[0].set_xlabel("$\Delta R$")
        
        # 2. Angular Resolution
        axes[1].hist(s_df['angle_diff'], bins=50, color='skyblue', edgecolor='black')
        axes[1].set_title("Angular Resolution (rad)"); axes[1].set_xlabel("$\Delta \Theta$")
        
        # 3. Norm Chi2
        axes[2].hist(s_df['norm_chi2'], bins=50, color='limegreen', edgecolor='black')
        axes[2].set_title("$\chi^2 / ndf$"); axes[2].set_xlabel("Value"); axes[2].set_yscale('log')

        # 4. Invariant Mass (신규 추가)
        # 유효한 질량값(0 이상)만 플로팅
        mass_data = s_df[s_df['inv_mass'] > 0]['inv_mass']
        axes[3].hist(mass_data, bins=50, color='orchid', edgecolor='black')
        axes[3].set_title("Invariant Mass (GeV)"); axes[3].set_xlabel("Mass")
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Parallel_Summary.png")

if __name__ == "__main__":
    main()