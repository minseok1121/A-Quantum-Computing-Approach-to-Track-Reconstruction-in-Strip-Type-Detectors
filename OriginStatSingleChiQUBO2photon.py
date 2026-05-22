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

import random
import matplotlib
matplotlib.use('Agg') 
from dwave.system import DWaveSampler, EmbeddingComposite
# --- [설정값: 모든 프로세스가 공유] ---
#STRIP_Z_LIST = [1.0, 1.7, 2.4, 4.4, 6.4, 8.4]
STRIP_Z_LIST = [1.3, 3.3, 5.3, 7.3, 9.3, 11.3, 13.3, 15.3, 17.3, 19.3]
STRIP_POS = np.linspace(-5.0 + (10.0/128)/2, 5.0 - (10.0/128)/2, 128)
PAD_SIZE = 1.0
OUTPUT_DIR = "QUBO_WOclustering_RealDoublePhoton_10lay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib.cm as cm # 구버전 호환용

# --- [기존 유틸리티 함수] ---
def to_stereo(x, y):
    inv_sqrt2 = 0.70710678
    return (x - y) * inv_sqrt2, (x + y) * inv_sqrt2

def plot_event_layers(csi_pool, ransac_pool, t_infos, event_id="Unknown"):
    """
    csi_pool: [[z, x, y, energy, type], ...]
    ransac_pool: [[z, x, y, 0, 0], ...]
    t_infos: MC Truth 정보 (pos, vec 포함)
    """
    # [변경] 2x5 레이아웃 (총 10개 레이어)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
    fig.suptitle(f"Event Display: CsI & RWELL with MC Truth (10 Layers, Event: {event_id})", fontsize=16)

    axes_flat = axes.flatten()

    if len(csi_pool) == 0 and len(ransac_pool) == 0:
        plt.close()
        return

    # 컬러맵 설정
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps['YlOrRd'].copy()
    except:
        import copy
        from matplotlib import cm
        cmap = copy.copy(cm.get_cmap('YlOrRd'))
    cmap.set_bad(color='white')
    
    csi_np = np.array(csi_pool) if len(csi_pool) > 0 else np.array([])
    rwell_np = np.array(ransac_pool) if len(ransac_pool) > 0 else np.array([])
    max_energy = csi_np[:, 3].max() if len(csi_np) > 0 else 1.0

    # [변경] 루프 범위를 10으로 확대
    for i in range(10):
        ax = axes_flat[i]
        
        # STRIP_Z_LIST에 10개 이상의 데이터가 있는지 확인
        if i < len(STRIP_Z_LIST):
            curr_z = STRIP_Z_LIST[i]
            
            # --- [Part 1: CsI 에너지 플롯] ---
            if csi_np.size > 0:
                diff = curr_z - csi_np[:, 0]
                mask = (diff > 0) & (diff < 1.5)
                layer_csi = csi_np[mask]
                
                if len(layer_csi) > 0:
                    from matplotlib.collections import PatchCollection
                    import matplotlib.patches as patches
                    rects = [patches.Rectangle((row[1]-0.5, row[2]-0.5), 1.0, 1.0) for row in layer_csi]
                    pc = PatchCollection(rects, cmap=cmap, alpha=0.7, edgecolors='gray', linewidths=0.5)
                    pc.set_array(layer_csi[:, 3])
                    pc.set_clim(0, max_energy)
                    ax.add_collection(pc)

            # --- [Part 2: RWELL 히트 플롯] ---
            if rwell_np.size > 0:
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

            ax.set_title(f"L{i+1} (z={curr_z:.1f}cm)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
        else:
            ax.axis('off') # Z 리스트보다 인덱스가 크면 빈 칸 처리

        # 공통 축 설정
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        if i >= 5: ax.set_xlabel("X [cm]") # 아래쪽 행
        if i % 5 == 0: ax.set_ylabel("Y [cm]") # 왼쪽 열

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/event_{event_id.replace('.root', '')}_10layers.png")
    plt.close()

def plot_debug_event(evt_id, ransac_pool, kf_pts_labeled, t_infos, errors):
    """
    [변경] 디버그 플롯도 10개 레이어(2x5)로 확장
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    reco_colors = ['red', 'green', 'orange', 'purple'] # PID가 많아질 경우 대비
    truth_colors = ['blue', 'cyan', 'magenta', 'navy']

    for i in range(10):
        ax = axes[i]
        if i >= len(STRIP_Z_LIST):
            ax.axis('off')
            continue
            
        z_s = STRIP_Z_LIST[i]
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        
        # 1. RANSAC Pool
        pool_layer = [p for p in ransac_pool if p[0] == z_s]
        if pool_layer:
            pool_layer = np.array(pool_layer)
            ax.scatter(pool_layer[:,1], pool_layer[:,2], color='gray', alpha=0.3, s=30, label='Candidates')

        # 2. MC Truth
        for t_idx, t_info in enumerate(t_infos):
            tx = t_info['x'] + (t_info['px']/t_info['pz']) * z_s
            ty = t_info['y'] + (t_info['py']/t_info['pz']) * z_s
            ax.scatter(tx, ty, color=truth_colors[t_idx % len(truth_colors)], facecolors='none', 
                       edgecolors=truth_colors[t_idx % len(truth_colors)], s=150, 
                       label=f'Truth {t_idx}' if i==0 else "")

        # 3. Reco Hits
        layer_reco = kf_pts_labeled[np.abs(kf_pts_labeled[:,0] - z_s) < 0.01]
        for row in layer_reco:
            pid = int(row[3])
            ax.scatter(row[1], row[2], color=reco_colors[pid % len(reco_colors)], marker='x', 
                       s=100, linewidths=2, label=f'Reco {pid}' if i==0 else "")

        ax.set_title(f"L{i+1} (z={z_s:.1f})", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0: ax.legend(loc='upper right', fontsize='xx-small')

    # --- 수정된 하단 에러 문자열 출력 세션 ---
    err_str = "\n".join([
        # e[3]을 지우고, e[2] 자리에 있던 inv_mass를 명시적으로 매칭
        f"Trk{k}: Dist={e[0]:.3f}, Ang={e[1]:.3f} | InvMass={e[2]:.3f}" 
        for k, e in errors.items()
    ])
    
    plt.suptitle(f"Double Photon Debug [{evt_id}] - 10 Layers\n{err_str}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Debug_{evt_id}_10layers.png")
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
    n_layers = len(layers_candidates)
    if n_layers < 3:
        return []

    print("\n" + "=" * 60)
    print("🔍 STRICT 2-SEGMENT CHAIN QUBO (EDGE VERSION)")
    print("=" * 60)

    # =========================
    # EDGE VARIABLES
    # x[l, i, j] = layer l i -> layer l+1 j
    # =========================
    edge_to_var = {}
    var_to_edge = {}

    vid = 0
    for l in range(n_layers - 1):
        for i in range(len(layers_candidates[l])):
            for j in range(len(layers_candidates[l + 1])):

                edge_to_var[(l, i, j)] = vid
                var_to_edge[vid] = (l, i, j)
                vid += 1

    print(f"\n🧩 Total EDGE Variables: {vid}")

    Q = {}

    def add_q(a, b, w):
        key = tuple(sorted((a, b)))
        Q[key] = Q.get(key, 0.0) + float(w)

    # =========================================================
    # EXACTLY-ONE
    # =========================================================

    A = 1000.0

    for l in range(n_layers - 1):

        edges = [
            edge_to_var[(l, i, j)]
            for i in range(len(layers_candidates[l]))
            for j in range(len(layers_candidates[l + 1]))
        ]

        for e in edges:
            add_q(e, e, -1.0 * A)

        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                add_q(edges[i], edges[j], 2.0 * A)

    # =========================================================
    # 2. STRICT CHAIN CONSISTENCY (FIXED)
    # =========================================================
    B = 500.0

    for j in range(len(layers_candidates[1])):

        incoming = [
            edge_to_var[(0, i, j)]
            for i in range(len(layers_candidates[0]))
        ]

        outgoing = [
            edge_to_var[(1, j, k)]
            for k in range(len(layers_candidates[2]))
        ]

        for a in incoming:
            for b in outgoing:
                add_q(a, b, -B)
    # =========================================================
    # 3. STRAIGHTNESS CONSTRAINT (FIXED FOR NODE STRUCTURE)
    # =========================================================
    C = 10.0

    for l in range(n_layers - 2):

        for i in range(len(layers_candidates[l])):
            for j in range(len(layers_candidates[l + 1])):
                for k in range(len(layers_candidates[l + 2])):

                    e1 = edge_to_var[(l, i, j)]
                    e2 = edge_to_var[(l + 1, j, k)]

                    x1, y1 = layers_candidates[l][i]['center']
                    x2, y2 = layers_candidates[l + 1][j]['center']
                    x3, y3 = layers_candidates[l + 2][k]['center']

                    dx1 = x2 - x1
                    dy1 = y2 - y1

                    dx2 = x3 - x2
                    dy2 = y3 - y2

                    bend = (dx1 - dx2)**2 + (dy1 - dy2)**2

                    add_q(e1, e2, C * bend)
   # =========================================================
    # DEBUG (LOWEST ENERGY BINARY TERMS + COORDS)
    # =========================================================

    binary = [(v, i, j) for (i, j), v in Q.items() if i != j]

    binary.sort(key=lambda x: x[0])  # lowest first

    print("\n" + "=" * 60)
    print("🔮 QUBO DEBUG (WITH COORDINATES)")
    print(f"Q Terms     : {len(Q)}")
    print(f"Edge Vars   : {vid}")
    print("=" * 60)

    print("\n📌 TOP 5 LOWEST BINARY TERMS")
    print("=" * 60)

    for w, i, j in binary[:5]:

        print(f"\nweight = {w:.3f}")

        # decode edges
        if i in var_to_edge and j in var_to_edge:

            l1, a, b = var_to_edge[i]
            l2, c, d = var_to_edge[j]

            p1 = layers_candidates[l1][a]['center']
            p2 = layers_candidates[l1 + 1][b]['center']

            p3 = layers_candidates[l2][c]['center']
            p4 = layers_candidates[l2 + 1][d]['center']

            print(f"edge1  = L{l1}: {a}->{b}  {p1} -> {p2}")
            print(f"edge2  = L{l2}: {c}->{d}  {p3} -> {p4}")

        else:
            print(f"vars   = ({i}, {j})")

        print("-" * 40)

    print("=" * 60)

    # =========================================================
    # SAMPLING
    # =========================================================
    sampleset = _SAMPLER.sample_qubo(
        Q,
        annealing_time=50,
        num_reads=100
    )

    # =========================================================
    # MINIMAL POST-SOLVE ANALYSIS + PATH
    # =========================================================

    best = sampleset.first
    sample = best.sample
    energy = best.energy

    print("\n" + "=" * 60)
    print("🧠 BEST SOLUTION SUMMARY")
    print("=" * 60)

    print(f"Best energy = {energy:.3f}")

    active = [v for v, val in sample.items() if val == 1]

    print(f"Active vars = {len(active)}")

    # =========================================================
    # PER-LAYER SUMMARY (NO FULL DUMP)
    # =========================================================

    path = [None] * n_layers

    for l in range(n_layers - 1):

        edges = [
            edge_to_var[(l, i, j)]
            for i in range(len(layers_candidates[l]))
            for j in range(len(layers_candidates[l + 1]))
        ]

        active_edges = [e for e in edges if sample.get(e, 0) == 1]

        print(f"\nLayer {l}->{l+1} | active={len(active_edges)}")

        for e in active_edges:

            ll, i, j = var_to_edge[e]

            p1 = layers_candidates[ll][i]
            p2 = layers_candidates[ll + 1][j]

            x1, y1 = p1['center']
            x2, y2 = p2['center']

            print(f"  {i}->{j}  ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f})")

        # path reconstruction seed
        if l == 0 and len(active_edges) > 0:
            _, i0, j0 = var_to_edge[active_edges[0]]
            path[0] = layers_candidates[0][i0]
            path[1] = layers_candidates[1][j0]

        if l == 1 and len(active_edges) > 0:
            _, _, j1 = var_to_edge[active_edges[0]]
            path[2] = layers_candidates[2][j1]

    # =========================================================
    # TOP ENERGY TERMS (WHY THIS WON)
    # =========================================================

    print("\n" + "=" * 60)
    print("⚖️ TOP ENERGY CONTRIBUTIONS")
    print("=" * 60)

    top_terms = []

    for (i, j), w in Q.items():

        xi = sample.get(i, 0)
        xj = sample.get(j, 0)

        if xi and xj:

            contrib = w * xi * xj

            top_terms.append((abs(contrib), i, j, w, contrib))

    top_terms.sort(reverse=True)

    for _, i, j, w, c in top_terms[:10]:

        print(f"({i},{j}) w={w:.1f} contrib={c:.1f}")
    
    # FINAL PATH
    # =========================================================

    print("\n" + "=" * 60)
    print("📌 FINAL PATH")
    print("=" * 60)

    for l, p in enumerate(path):

        if p is None:
            print(f"L{l}: NONE")
            continue

        x, y = p['center']
        print(f"L{l}: ({x:.3f},{y:.3f})")

    return path

def run_single_photon_tracking(cluster_pts, csi_pool, pid, file_label, iterations=10):
    csi_pool = np.asanyarray(csi_pool)
    cluster_pts = np.asanyarray(cluster_pts)
    
    unique_zs = np.sort(np.unique(cluster_pts[:, 0]))
    # 앞 최대 5개만 사용
    n_pick = min(5, len(unique_zs))

    # 그 안에서 랜덤 인덱스 3개 선택
    rand_idx = sorted(random.sample(range(n_pick), min(3, n_pick)))

    # 선택된 인덱스로 target_zs 구성
    target_zs = [unique_zs[i] for i in rand_idx]
    if len(target_zs) == 0: return None

    # --- 초기 관심 영역(ROI) 설정 (모든 힛을 포함하는 기하학적 중심으로 수정) ---
    current_rois = []
    for z in target_zs:
        layer_hits = cluster_pts[cluster_pts[:, 0] == z]
        is_stereo = z in [3.3, 7.3]
        
        if len(layer_hits) > 0:
            if is_stereo:
                # 45도 회전된 좌표계에서 범위를 계산
                rot_x, rot_y = to_stereo(layer_hits[:, 1], layer_hits[:, 2])
                
                # 모든 점을 포함하는 회전 좌표계 기준의 경계 계산
                min_rx, max_rx = np.min(rot_x), np.max(rot_x)
                min_ry, max_ry = np.min(rot_y), np.max(rot_y)
                
                width_x = max_rx - min_rx
                width_y = max_ry - min_ry
                
                # [수정] 평균(mean) 대신, 모든 점을 완벽히 감싸는 기하학적 중심(Center of Bounding Box) 계산
                l_cx_rot = (min_rx + max_rx) / 2.0
                l_cy_rot = (min_ry + max_ry) / 2.0
                
                # 원래 좌표계로 역회전 변환
                l_cx = (l_cx_rot + l_cy_rot) / (2 * 0.70710678)
                l_cy = (l_cy_rot - l_cx_rot) / (2 * 0.70710678)
            else:
                # 일반 좌표계 기준 경계 계산
                min_x, max_x = np.min(layer_hits[:, 1]), np.max(layer_hits[:, 1])
                min_y, max_y = np.min(layer_hits[:, 2]), np.max(layer_hits[:, 2])
                
                width_x = max_x - min_x
                width_y = max_y - min_y
                
                # [수정] 모든 점을 완벽히 감싸는 기하학적 중심 계산
                l_cx = (min_x + max_x) / 2.0
                l_cy = (min_y + max_y) / 2.0
            
            raw_range = max(width_x, width_y)/2
            # 초기 사이즈를 꽉 채우는 범위(raw_range)에 마진을 살짝 더해주면(예: * 1.1) 더욱 안전합니다.
            dynamic_size = raw_range / 0.75 + 0.0 * len(layer_hits)
        else:
            dynamic_size = 2.0
            # 레이어에 힛이 아예 없는 경우 전체 점의 기하학적 영역 중심을 대입
            l_cx = (np.min(cluster_pts[:, 1]) + np.max(cluster_pts[:, 1])) / 2.0
            l_cy = (np.min(cluster_pts[:, 2]) + np.max(cluster_pts[:, 2])) / 2.0

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
            is_stereo = z in [3.3, 7.3]
            
            layer_hits_all = csi_pool[csi_pool[:, 0] == z]
            layer_cluster_hits = cluster_pts[cluster_pts[:, 0] == z]
            next_s = s * 0.75 if it < iterations - 1 else s
            
            # =================================================================
            # ✨ [완전 단순화] 데이터 기반 랜덤 무게중심 수렴 (Mean-Shift)
            # =================================================================
            # 1. 원래 할당받은 중심점(rcx, rcy) 기준으로 주변에 30개 랜덤 흩뿌리기
            candidate_centers = [(rcx, rcy)]
            n_random_nodes = 20
            max_radius = next_s * 1.2
            
            for _ in range(n_random_nodes):
                theta = np.random.uniform(0, 2 * np.pi)
                r = max_radius * np.sqrt(np.random.uniform(0, 1)) # Uniform Disk Sampling
                
                n_cx = rcx + r * np.cos(theta)
                n_cy = rcy + r * np.sin(theta)
                candidate_centers.append((n_cx, n_cy))
            
            # 2. 흩뿌려진 후보들 내부의 힛트 무게중심으로 좌표 워프 및 최종 노드 확정
            sub_nodes = []
            total_layer_energy = np.sum(layer_hits_all[:, 3]) if len(layer_hits_all) > 0 else 1.0
            n_total_hits = len(layer_cluster_hits)
            
            # 중복 위치 방지를 위한 최소 거리 기준 (영역 크기의 10% 수준)
            duplicate_threshold = next_s * 0.5
            
            for c_x, c_y in candidate_centers:
                # 현재 후보지 영역(next_s) 내에 들어오는 힛트 필터링 (Stereo 여부 반영)
                if is_stereo:
                    cls_rot_x, cls_rot_y = to_stereo(layer_cluster_hits[:, 1], layer_cluster_hits[:, 2])
                    n_rot_cx, n_rot_cy = to_stereo(c_x, c_y)
                    cluster_mask = (np.abs(cls_rot_x - n_rot_cx) < next_s/2) & (np.abs(cls_rot_y - n_rot_cy) < next_s/2)
                else:
                    cluster_mask = (np.abs(layer_cluster_hits[:, 1] - c_x) < next_s/2) & (np.abs(layer_cluster_hits[:, 2] - c_y) < next_s/2)
                
                # [핵심] 영역 내에 힛트가 있다면, 그 힛트들의 평균 위치(무게중심)로 중심점을 강제 워프!
                if np.any(cluster_mask):
                    hits_in_area = layer_cluster_hits[cluster_mask]
                    final_cx = np.mean(hits_in_area[:, 1])
                    final_cy = np.mean(hits_in_area[:, 2])
                else:
                    # 주변에 힛트가 하나도 안 걸리는 허공 후보지는 패스
                    continue
                
                # 🚨 [스마트 중복 제거] 이미 등록된 sub_nodes 중에 방금 계산한 무게중심과 너무 가까운 녀석이 있는지 검사
                is_duplicate = False
                for existing_node in sub_nodes:
                    ex_cx, ex_cy = existing_node['center']
                    dist = np.sqrt((final_cx - ex_cx)**2 + (final_cy - ex_cy)**2)
                    if dist < duplicate_threshold:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue # 이미 이 데이터 뭉치에는 깃발이 꽂혔으니 이번 랜덤 점은 무시하고 패스!
                
                # 3. 워프된 '고유한' 진짜 데이터 정중앙(final_cx, final_cy) 기준으로 최종 스코어 계산
                if is_stereo:
                    all_rot_x, all_rot_y = to_stereo(layer_hits_all[:, 1], layer_hits_all[:, 2])
                    f_rot_cx, f_rot_cy = to_stereo(final_cx, final_cy)
                    energy_mask = (np.abs(all_rot_x - f_rot_cx) < next_s/2) & (np.abs(all_rot_y - f_rot_cy) < next_s/2)
                else:
                    energy_mask = (np.abs(layer_hits_all[:, 1] - final_cx) < next_s/2) & (np.abs(layer_hits_all[:, 2] - final_cy) < next_s/2)
                
                area_energy = np.sum(layer_hits_all[energy_mask, 3]) if np.any(energy_mask) else 0
                n_hits_in_area = np.sum(cluster_mask)
                
                score = (area_energy/total_layer_energy + n_hits_in_area/n_total_hits) / np.log1p(n_total_hits) + 0.001
                sub_nodes.append({'center': (final_cx, final_cy), 'size': next_s, 'score': score})
            
            # 만약 난수가 몽땅 허공을 찔러 아무도 못 잡았다면, 최소한의 백업만 유지
            if not sub_nodes:
                sub_nodes.append({'center': (rcx, rcy), 'size': next_s, 'score': 0.0001})
                
            layers_candidates.append(sub_nodes)

        all_combinations = []
        
        # 3개 레이어의 모든 노드 조합을 완전 탐색 (후보가 레이어당 30개 내외라 순식간에 계산 가능)
        if len(layers_candidates) == len(target_zs):
            import itertools
            # 각 레이어의 노드 인덱스 조합 생성
            node_indices = [range(len(nodes)) for nodes in layers_candidates]
            
            for idx_triple in itertools.product(*node_indices):
                pts = []
                for l_idx, n_idx in enumerate(idx_triple):
                    cx, cy = layers_candidates[l_idx][n_idx]['center']
                    cz = target_zs[l_idx]
                    pts.append([cx, cy, cz])
                pts = np.array(pts) # Shape: (3, 3) -> 각 행은 [x, y, z]
                
                # 3차원 공간에서 3개 점의 직선성 오차 계산 (간단한 3D 피팅 잔차 사용)
                # Z축 기준 X, Y 각각 선형 회귀 후 오차 제곱합 계산
                A = np.vstack([pts[:, 2], np.ones(len(pts))]).T
                # X 오차
                m_x, c_x = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]
                res_x = np.sum((pts[:, 0] - (m_x * pts[:, 2] + c_x)) ** 2)
                # Y 오차
                m_y, c_y = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
                res_y = np.sum((pts[:, 1] - (m_y * pts[:, 2] + c_y)) ** 2)
                
                total_residual = res_x + res_y
                all_combinations.append({
                    'indices': idx_triple,
                    'residual': total_residual,
                    'points': pts[:, :2] # 시각화에 쓸 (x, y) 좌표들만 저장
                })
                
            # 직선성 오차(residual)가 작은 순서대로 정렬 (가장 완벽한 직선이 1등)
            all_combinations = sorted(all_combinations, key=lambda x: x['residual'])

        # 3. 시각화 및 상위 순위쌍 표시
        top_colors = ['gold', 'limegreen', 'darkorange']  # 1등(금), 2등(초록), 3등(주황)
        top_markers = ['o', '*', '^']                     # 마커 모양 차별화
        
        for i, z in enumerate(target_zs):
            ax = axes[i]
            layer_cluster_hits = cluster_pts[cluster_pts[:, 0] == z]
            sub_nodes = layers_candidates[i]
            is_stereo = z in [3.3, 7.3]
            
            # 🚨 [수정] 이 레이어(i) 이터레이션의 진짜 기준점(중심)을 다시 정확하게 매칭합니다.
            roi = current_rois[i]
            layer_rcx, layer_rcy = roi['center']
            
            # 기본 힛트 및 후보 그물망 사각형 그리기 (기존 로직 유지)
            ax.scatter(layer_cluster_hits[:, 1], layer_cluster_hits[:, 2], color='blue', alpha=0.3)
            for node in sub_nodes:
                if is_stereo:
                    rect = plt.Rectangle((node['center'][0] - node['size']/2, node['center'][1] - node['size']/2), 
                                        node['size'], node['size'], fill=False, color='red', alpha=0.3)
                    trans = mtransforms.Affine2D().rotate_around(node['center'][0], node['center'][1], np.deg2rad(45)) + ax.transData
                    rect.set_transform(trans)
                    ax.add_patch(rect)
                else:
                    ax.add_patch(plt.Rectangle((node['center'][0]-node['size']/2, node['center'][1]-node['size']/2), 
                                                node['size'], node['size'], fill=False, color='gray', alpha=0.3))
            
            # 🚨 [수정] 위에서 가져온 레이어 고유의 기준점(layer_rcx, layer_rcy)으로 x 표시를 찍습니다.
            ax.scatter(layer_rcx, layer_rcy, marker='x', color='black', s=60, zorder=20, linewidths=1.5)
            
            # 🌟 여기에 해당 레이어에 속한 Top 3 순위쌍의 노드 중심점을 오버레이해서 그리기
            for rank in range(min(3, len(all_combinations))):
                best_combo = all_combinations[rank]
                # 이번 레이어(i)에 해당하는 점의 (x, y) 추출
                best_x, best_y = best_combo['points'][i]
                
                ax.scatter(best_x, best_y, marker=top_markers[rank], color=top_colors[rank], 
                        s=180, zorder=15, edgecolors='black', linewidths=2,
                        label=f'Rank {rank+1} Line' if i == 0 else "") # 범례는 첫 창에만
                
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            if i == 0 and len(all_combinations) > 0:
                ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout()
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
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
    #xy_coords = pts[:, 1:3]
    #kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(xy_coords)
    #labels = kmeans.labels_
    labels = np.zeros(len(pts))
    # n_components는 클러스터 개수(K)와 같습니다.
    #gmm = GaussianMixture(n_components=2, random_state=42).fit(xy_coords)
    #labels = gmm.predict(xy_coords)

    all_tracks_pts = []
    all_slopes = []

    # --- [STEP 2] 각 클러스터(포톤)에 대해 독립적으로 Reco 실행 ---
    for pid in range(10):
        # 해당 클러스터에 할당된 힛들만 필터링
        cluster_hits = pts
        
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
            CSI_Z_BOUNDS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)] # CsI 층의 z 범위 (예시)
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

                kf_pts_labeled, slopes = results
                track_errors = {}
                matches = []

                # =========================================================
                # 1. SMART TRACK MATCHING (FULL TRACK RESIDUAL)
                # =========================================================

                match_candidates = []

                for r_idx in range(len(slopes)):

                    reco_pts = kf_pts_labeled[kf_pts_labeled[:, 3] == r_idx]

                    if len(reco_pts) == 0:
                        continue

                    for t_idx, t_info in enumerate(t_infos):

                        total_cost = 0.0

                        for pt in reco_pts:

                            pz, px, py = pt[0], pt[1], pt[2]

                            tx = t_info['x'] + (t_info['px'] / t_info['pz']) * pz
                            ty = t_info['y'] + (t_info['py'] / t_info['pz']) * pz

                            total_cost += (px - tx)**2 + (py - ty)**2

                        match_candidates.append({
                            'r_idx': r_idx,
                            't_idx': t_idx,
                            'cost': total_cost
                        })

                # =========================================================
                # GREEDY 1:1 MATCH
                # =========================================================

                match_candidates.sort(key=lambda x: x['cost'])

                assigned_reco = set()
                assigned_truth = set()

                for cand in match_candidates:

                    if cand['r_idx'] not in assigned_reco and cand['t_idx'] not in assigned_truth:

                        matches.append((cand['r_idx'], cand['t_idx']))

                        assigned_reco.add(cand['r_idx'])
                        assigned_truth.add(cand['t_idx'])

                # =========================================================
                # KEEP ONLY MATCHED RECO TRACKS
                # =========================================================

                matched_ridx = set(r for r, _ in matches)

                kf_pts_labeled = np.array([
                    pt for pt in kf_pts_labeled
                    if int(pt[3]) in matched_ridx
                ])

                # =========================================================
                # ERROR / CHI2 / ANGLE
                # =========================================================

                reco_results = {}

                for r_idx, t_idx in matches:

                    reco_pts_subset = kf_pts_labeled[kf_pts_labeled[:, 3] == r_idx]

                    if len(reco_pts_subset) == 0:
                        continue

                    rsx, rsy = slopes[r_idx]

                    rz0, rx0, ry0 = reco_pts_subset[0, 0], reco_pts_subset[0, 1], reco_pts_subset[0, 2]

                    best_t = t_infos[t_idx]

                    # -----------------------------------------------------
                    # (A) DISTANCE AT Z=0
                    # -----------------------------------------------------

                    tx_at_z0 = best_t['x']
                    ty_at_z0 = best_t['y']

                    rx_at_z0 = rx0 + rsx * (0 - rz0)
                    ry_at_z0 = ry0 + rsy * (0 - rz0)

                    dist = np.sqrt((rx_at_z0 - tx_at_z0)**2 + (ry_at_z0 - ty_at_z0)**2)

                    # -----------------------------------------------------
                    # (B) ANGLE DIFFERENCE
                    # -----------------------------------------------------

                    r_vec = np.array([rsx, rsy, 1.0])
                    r_vec /= np.linalg.norm(r_vec)

                    t_vec = best_t['vec']

                    angle = np.arccos(np.clip(np.dot(t_vec, r_vec), -1.0, 1.0))

                    # -----------------------------------------------------
                    # (C) REDUCED CHI2
                    # -----------------------------------------------------

                    chi2_sum = 0.0

                    for pt in reco_pts_subset:

                        pz, px, py = pt[0], pt[1], pt[2]

                        exp_x = best_t['x'] + (best_t['px'] / best_t['pz']) * pz
                        exp_y = best_t['y'] + (best_t['py'] / best_t['pz']) * pz

                        chi2_sum += (px - exp_x)**2 + (py - exp_y)**2

                    ndf = len(reco_pts_subset) * 2

                    reduced_chi2 = chi2_sum / ndf if ndf > 0 else 999

                    reco_results[r_idx] = {
                        'vec': r_vec,
                        'energy': best_t['energy'],
                        'dist': dist,
                        'angle': angle,
                        'chi2': reduced_chi2
                    }

                # =========================================================
                # INVARIANT MASS
                # =========================================================

                inv_mass = -1.0

                if len(reco_results) >= 2:

                    keys = list(reco_results.keys())

                    r1 = reco_results[keys[0]]
                    r2 = reco_results[keys[1]]

                    cos_theta = np.clip(np.dot(r1['vec'], r2['vec']), -1.0, 1.0)

                    inv_mass = np.sqrt(2 * r1['energy'] * r2['energy'] * (1 - cos_theta))

                # =========================================================
                # SAVE STATS
                # =========================================================

                for r_idx, info in reco_results.items():

                    track_errors[r_idx] = (info['dist'], info['angle'], inv_mass)

                    local_stats.append({
                        'dist_start': info['dist'],
                        'angle_diff': info['angle'],
                        'norm_chi2': info['chi2'],
                        'inv_mass': inv_mass
                    })

                # =========================================================
                # DEBUG PLOT
                # =========================================================

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
    file_list = glob.glob("/gv0/Users/mioh/DAMSA_FULL/ms/ALPGun/ALP100DoublePhoton_0_05rad_10Layers/DAMSA_m100_r1*.root")
    
    # [추가] 결과 저장용 파일 경로
    PICKLE_PATH = f"{OUTPUT_DIR}/analysis_results.pkl"
    
    # 만약 이미 파일이 있다면 로드할지 결정 (선택 사항)
    if os.path.exists(PICKLE_PATH):
         with open(PICKLE_PATH, 'rb') as f:
             results = pickle.load(f)
         print(f">>> 기존 피클 파일을 로드했습니다: {PICKLE_PATH}")
    else:
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
        
        def get_overflow_adjusted_data(data, lower_p=0, upper_p=90):
            """
            데이터의 상위 경계값을 넘는 Overflow 데이터를 
            마지막 Bin의 위치로 뭉쳐주는 헬퍼 함수
            """
            if data.empty:
                return data, 0.0, 1.0
            
            # 0% ~ 90% 경계값 계산
            v_min, v_max = np.percentile(data, [lower_p, upper_p])
            
            # 물리량 특성상 하한선이 0 미만으로 내려가지 않도록 잠금 (Invariant Mass 제외 유연하게 적용 가능)
            if v_min < 0 and data.min() >= 0:
                v_min = 0.0
                
            # 안전장치: 모든 데이터가 같아서 v_min과 v_max가 같은 경우 빈 폭 강제 확보
            if v_min == v_max:
                v_max = v_min + 1.0
                
            # v_max를 넘어가는(Overflow) 값들을 v_max 직전 값으로 클리핑하여 마지막 빈에 모이게 함
            # np.clip을 사용하면 v_max boundary에 걸쳐 마지막 bin에 누적됩니다.
            clipped_data = np.clip(data, v_min, v_max)
            
            return clipped_data, v_min, v_max

        # ----------------------------------------------------
        # 1. Vertex Resolution
        # ----------------------------------------------------
        v_data = s_df['dist_start'].dropna()
        if not v_data.empty:
            v_clip, v_min, v_max = get_overflow_adjusted_data(v_data, 0, 90)
            axes[0].hist(v_clip, bins=50, range=(v_min, v_max), color='salmon', edgecolor='black')
            axes[0].set_xlim(v_min, v_max * 1.05)  # Overflow 빈이 벽에 딱 붙지 않게 약간의 여백 제공
        else:
            axes[0].hist([], bins=50, color='salmon', edgecolor='black')
        axes[0].set_title("Vertex Resolution (cm)\n[Last Bin: Upper 10% Overflow]"); axes[0].set_xlabel("$\Delta R$")
        
        # ----------------------------------------------------
        # 2. Angular Resolution
        # ----------------------------------------------------
        a_data = s_df['angle_diff'].dropna()
        if not a_data.empty:
            a_clip, a_min, a_max = get_overflow_adjusted_data(a_data, 0, 90)
            axes[1].hist(a_clip, bins=50, range=(a_min, a_max), color='skyblue', edgecolor='black')
            axes[1].set_xlim(a_min, a_max * 1.05)
        else:
            axes[1].hist([], bins=50, color='skyblue', edgecolor='black')
        axes[1].set_title("Angular Resolution (rad)\n[Last Bin: Upper 10% Overflow]"); axes[1].set_xlabel("$\Delta \Theta$")
        
        # ----------------------------------------------------
        # 3. Norm Chi2
        # ----------------------------------------------------
        c_data = s_df['norm_chi2'].dropna()
        if not c_data.empty:
            c_clip, c_min, c_max = get_overflow_adjusted_data(c_data, 0, 90)
            axes[2].hist(c_clip, bins=50, range=(c_min, c_max), color='limegreen', edgecolor='black')
            axes[2].set_xlim(c_min, c_max * 1.05)
            axes[2].set_yscale('log')  # 카이제곱 특성상 로그스케일 유지
        else:
            axes[2].hist([], bins=50, color='limegreen', edgecolor='black')
        axes[2].set_title("$\chi^2 / ndf$\n[Last Bin: Upper 10% Overflow]"); axes[2].set_xlabel("Value")

        # ----------------------------------------------------
        # 4. Invariant Mass
        # ----------------------------------------------------
        mass_data = s_df[s_df['inv_mass'] > 0]['inv_mass'].dropna()
        if not mass_data.empty:
            m_clip, m_min, m_max = get_overflow_adjusted_data(mass_data, 0, 90)
            axes[3].hist(m_clip, bins=50, range=(m_min, m_max), color='orchid', edgecolor='black')
            axes[3].set_xlim(m_min, m_max * 1.05)
        else:
            axes[3].hist([], bins=50, color='orchid', edgecolor='black')
        axes[3].set_title("Invariant Mass (GeV)\n[Last Bin: Upper 10% Overflow]"); axes[3].set_xlabel("Mass")
        
        # ----------------------------------------------------
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Parallel_Summary.png")

if __name__ == "__main__":
    main()