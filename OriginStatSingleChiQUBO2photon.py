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
            if not (len(t_df) >= 2 and (t_df['energy'] >= 500).all()):
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
            results = apply_two_photon_split_reco(ransac_pool, csi_pool, f_path.split('/')[-1])

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

def main():
    file_list = glob.glob("/gv0/Users/mioh/DAMSA_FULL/ms/ALPGun/ALP100DoublePhotonOrigin/DAMSA_m100_r*.root")
    
    # [추가] 결과 저장용 파일 경로
    PICKLE_PATH = f"{OUTPUT_DIR}/analysis_results.pkl"
    
    # 만약 이미 파일이 있다면 로드할지 결정 (선택 사항)
    # if os.path.exists(PICKLE_PATH):
    #     with open(PICKLE_PATH, 'rb') as f:
    #         results = pickle.load(f)
    #     print(f">>> 기존 피클 파일을 로드했습니다: {PICKLE_PATH}")
    # else:
    
    results = []
    for f in file_list:
        print(f">>> 파일 처리 중: {f}")
        res = process_single_file(f) 
        results.append(res)
        plt.close('all') 
        gc.collect()

    # --- [핵심: 피클 저장] ---
    # results에는 (total_processed, total_passed, combined_stats) 튜플들이 들어있음
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 모든 결과가 피클로 저장되었습니다: {PICKLE_PATH}")

    # [Step 3] 통계 취합 및 플로팅
    total_processed = sum(r[0] for r in results if r)
    total_passed = sum(r[1] for r in results if r)
    combined_stats = []
    for r in results: 
        if r: combined_stats.extend(r[2])

    print(f"📊 분석 요약: 총 {total_processed} 중 {total_passed} 이벤트 선택됨")

    if combined_stats:
        plt.close('all')
        s_df = pd.DataFrame(combined_stats)
        
        # 데이터프레임 자체도 나중에 쓰기 편하게 CSV나 별도 피클로 저장 가능
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