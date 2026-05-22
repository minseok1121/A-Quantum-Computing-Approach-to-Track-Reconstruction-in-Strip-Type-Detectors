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
