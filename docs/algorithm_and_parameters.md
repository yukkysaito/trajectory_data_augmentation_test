# 軌跡データオーグメンテーション: アルゴリズムとパラメータ解説

このドキュメントは、本リポジトリが実装する Diffusion Planner 向け軌跡オーグメンテーションの
アルゴリズム解説と、全パラメータのリファレンスをまとめたものです。

- 対象コード: `trajectory_augmentation/core.py`(ロジック)、`trajectory_augmentation/cli.py`(CLI)、
  `trajectory_augmentation/visualization.py`(Plotly 可視化)

---

## 1. 目的と背景

自動運転の学習ベースプランナーは、GT(ground truth)軌跡だけで模倣学習すると
「経路からずれた状態からの復帰」を学習できません。本実装は、横方向の
トラッキング誤差・自己位置推定誤差を模擬するため、現在時刻 `t0` の自車姿勢を
横に `offset` [m]・方位を `yaw_offset` [deg] ずらし、その前後を滑らかな
ブリッジで接続した拡張軌跡を生成します。

- **過去ブリッジ(M 秒)**: 元の過去軌跡 → ずらした現在姿勢
- **未来ブリッジ(N 秒)**: ずらした現在姿勢 → 元の GT 未来軌跡(復帰)

### ショートカット学習(リーク)対策

過去ブリッジは「離脱の形」を履歴に符号化するため、モデルが地図を見ずに
自車履歴の外挿だけで復帰軌道を当てられてしまう恐れがあります(模倣学習の
ショートカット/copycat 問題)。これを緩和するために 2 つの仕組みがあります。

1. **M/N のランダム化**: 最小実行可能なブリッジ時間の上に乱数で上乗せし、
   同じ過去に複数の復帰が対応するようにする
2. **過去履歴バンプ**: 過去(条件付け側)にのみランダムな横方向の「山」を
   1 つ注入し、綺麗な離脱シグネチャを壊す。未来ターゲットには触れない

---

## 2. 時間ホライズン

| 区間 | 範囲 | 用途 |
|---|---|---|
| フル GT ホライズン | 過去 5 s / 未来 10 s | 計算に使用。大きなオフセットでもブリッジ用の余裕を確保 |
| 出力ウィンドウ | 過去 3 s / 未来 8 s | 学習データとして切り出す範囲。プロットの主表示範囲 |

サンプリング周期は `dt = 0.1 s`。`t = 0` が現在時刻 `t0`(`current_index`)です。

---

## 3. アルゴリズム

### 3.1 テストパターン生成(`generate_synthetic_gt`)

形状 3 種 × 速度プロファイル 5 種 = 15 パターンの合成 GT を生成し、
`test_patterns/*.csv` に保存します。

**形状**(曲率プロファイル `build_curvature_profile`、`s_rel` は t0 基準の弧長):

- `straight`: κ = 0
- `curve`: κ = 0.010 + 0.0015 sin(0.030 s_rel + 0.2)(緩い定常旋回)
- `s_curve`: κ = ±0.014 のガウシアン対(S 字)

**速度プロファイル**(`build_speed_profile`):

- `constant`: 8 m/s 一定
- `decelerating`: 10 → 6 m/s(smoothstep)
- `accelerating`: 6 → 10 m/s(smoothstep)
- `stopping`: t0 まで 8 m/s、t = 5 s で停止
- `stop8s`: t = 6 s から減速し t = 8 s で停止

曲率と速度を台形則で積分して (x, y, yaw) を求め、t0 原点・t0 方位基準に正規化します。

### 3.2 センターライン化と横オフセットプロファイル

対象セグメント(未来、または時間反転した過去)を弧長 0.05 m 間隔の
センターラインに稠密化し(`build_centerline`)、横オフセットを弧長 s の
**5 次多項式** `l(s)` で減衰させます(`solve_lateral_profile_coeffs`)。

境界条件(合流点の弧長を L = `s_merge` として):

```
l(0) = offset          l(L) = 0
l'(0) = tan(yaw_offset) l'(L) = 0
l''(0) = 0              l''(L) = 0
```

これにより開始点で指定のオフセット・方位ずれを持ち、合流点で位置・接線・
曲率が連続(C2)になります。パスはセンターライン + 法線方向 × l(s) で構成
します(`build_merge_path`)。

### 3.3 合流点の探索(`plan_recovery_path` / `solve_merge_centerline_s`)

ブリッジ時間 `connect_time_s` を GT の走行距離に変換したものを距離バジェット
とし、オフセットパスの実弧長がバジェット内に収まる最大の合流点 `s_merge` を
二分法で探索します。オフセットパスは元より長くなるため、この調整をしないと
速度が不自然に上がります。ごく短いブリッジでは代わりに速度スケール
(`connect_speed_scale`)で吸収します。

### 3.4 進捗ベースの速度整合(`augment_directed_segment`)

拡張パス上の位置は「時刻 → 進捗(弧長)」のマッピング(`progress_profile`)で
決めます。ブリッジ以降は GT の時刻-距離関係を合流点基準でシフトして流用する
ため、**拡張軌跡の速度プロファイルは GT と同じ意味を保ちます**。
`exact_speed_profile` は進捗の時間微分として厳密に計算されます。

### 3.5 双方向化(`augment_trajectory_bidirectional`)

過去側は時間反転したセグメント(`extract_reversed_past_segment`)として
未来側と同じロジックに通します。進行方向が逆になるため **方位オフセットは
符号反転**して渡します。最後に過去・未来を結合し、yaw は結合後の幾何から
再計算(`compute_forward_yaw`)して全チャネルの整合を取ります。

### 3.6 制約診断(`evaluate_constraints_in_window`)

出力ウィンドウ内で 3 つの物理制約を評価します。

| 制約 | 定義 | 評価範囲 | デフォルト上限 |
|---|---|---|---|
| 横加速度 | a_lat = v² κ(v は厳密弧長速度、κ はサンプル軌跡の曲率) | 出力ウィンドウ全体 | 3.0 m/s² |
| ブリッジ速度乖離 | \|v_aug − v_gt\| | ブリッジ区間 [−M, +N] | 0.5 m/s |
| ブリッジ縦ジャーク | 加速度の時間微分の絶対値 | ブリッジ区間 [−M, +N] | 5.0 m/s³ |

3 つすべて満たすと `passes = True` です。

### 3.7 実行可能性探索(`search_feasible_result`)

初期 M/N(デフォルト 0.1 s / 0.1 s)で制約違反がある場合、0.1 s 刻みで探索します。

1. **extend N**: M を固定し、N を伸ばして最小の合格値を探す(N ≤ 10 s)
2. **extend M and N**: それでも駄目なら (M, N) の組を総当たり(M ≤ 5 s)

見つからなければ「No feasible candidate」を報告します。

### 3.8 M/N ランダム化(`randomize_bridge_times_result`)

`--randomize-bridge-times` 指定時、最小実行可能な (M, N) を下限として

```
N ← snap( N_min + U(0, extra_range) )   # 0.1 s グリッドにスナップ
M ← snap( M_min + U(0, extra_range) )
```

をサンプリングし、制約を再検証します(不合格なら再抽選、最大 10 回)。
長いブリッジは通常なだらかですが、曲線区間では保証されないため毎回
チェックします。

### 3.9 過去履歴バンプ(`sample_past_history_bump` / `apply_lateral_bump_to_dense_path`)

`--past-bump` 指定時、過去履歴に **1 つ**の横方向バンプを注入します。

**バンプの形**(窓関数 `lateral_bump_window`):

```
b(u) = amplitude × sin³(πu),   u = (σ − σ_start) / (σ_end − σ_start) ∈ [0, 1]
```

sin³ は両端で値・1 階微分・2 階微分がすべてゼロになるため、周囲のパスと
C2 連続(曲率まで連続)で接続されます。

**サンプリング**(固定の内部定数から一様乱数):

```
duration_s   ← U(BUMP_DURATION_RANGE_S)              # 長さ
start_time_s ← U(BUMP_MIN_START_TIME_S, 3.0 − duration)  # 位置(t0 からの遡り)
amplitude_m  ← U(BUMP_AMPLITUDE_RANGE_M) × {−1, +1}   # 高さと左右
```

**適用手順**:

1. バンプの時間窓を進捗プロファイル経由で弧長窓 [σ_start, σ_end] に変換
2. 稠密パスの法線方向にオフセットを加算して幾何を摂動
3. 弧長・yaw を摂動後の幾何から再計算(チャネル整合性の維持)
4. 進捗を旧弧長 → 新弧長で再マップ(バンプの余剰長は局所的な速度上昇として現れる)
5. 制約を再検証。違反なら 3 つ組を引き直し(最大 20 回、全滅ならバンプなし)

バンプはブリッジ区間と重なってもよく、安全性は配置ルールではなく制約の
再評価で担保します。保護されるのは t0 直前(`BUMP_MIN_START_TIME_S`)のみです。
バンプは過去(条件付け)側専用で、未来ターゲットには適用されません。

### 3.10 処理フロー全体(`run_demo_case`)

```
GT 読み込み
  → 初期 M/N で双方向オーグメンテーション(シード軌跡)
  → 制約診断
  → [adaptive] 実行可能性探索(extend N → extend M and N)
  → [--randomize-bridge-times] M/N ランダム化 + 再検証
  → [--past-bump] バンプ注入 + 再検証
  → DemoArtifacts(シード軌跡・採用軌跡・診断)を返す
```

`--no-adaptive-bridge-search` の場合は探索・ランダム化・バンプをすべて
スキップし、指定 M/N の診断のみ行います。

---

## 4. パラメータリファレンス

### 4.1 モジュール定数(`core.py`)

| 定数 | 値 | 意味 |
|---|---|---|
| `FULL_PAST_HORIZON_S` / `FULL_FUTURE_HORIZON_S` | 5.0 / 10.0 | フル GT ホライズン [s] |
| `OUTPUT_PAST_HORIZON_S` / `OUTPUT_FUTURE_HORIZON_S` | 3.0 / 8.0 | 出力ウィンドウ [s] |
| `DEFAULT_DT` | 0.1 | サンプリング周期 [s]。探索のグリッド幅も兼ねる |
| `MIN_BRIDGE_TIME_S` | 0.1 | M/N 未指定時の探索開始値 [s] |
| `BUMP_AMPLITUDE_RANGE_M` | (0.05, 0.20) | バンプ振幅の一様サンプリング範囲 [m](符号は 50/50) |
| `BUMP_DURATION_RANGE_S` | (1.0, 2.0) | バンプ持続時間の一様サンプリング範囲 [s] |
| `BUMP_MIN_START_TIME_S` | 0.3 | t0 直前の保護マージン [s] |

バンプの範囲を変えたい場合はこの定数を直接編集します(CLI からは変更不可)。

### 4.2 CLI 引数(`diffusion_planner_augmentation.py`)

**必須(実行モード)**

| 引数 | 意味 |
|---|---|
| `--adaptive-bridge-search` / `--no-adaptive-bridge-search` | 自動ブリッジ時間探索の有効/無効。どちらか必須(`--list-patterns` / `--write-pattern-csvs` を除く) |

**基本**

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--pattern` | `s_curve_constant` | テストパターン名(形状_速度) |
| `--pattern-dir` | `test_patterns/` | パターン CSV の場所 |
| `--seed` | 7 | 乱数シード(オフセット・M/N・バンプの抽選に使用) |
| `--offset` | None(乱数: 0.4〜1.2 m × ±) | 横オフセット [m] |
| `--yaw-offset-deg` | 0.0 | 方位オフセット [deg] |
| `--recover-time` | None(0.1 s から探索) | 未来ブリッジ N [s] |
| `--past-connect-time` | None(0.1 s から探索) | 過去ブリッジ M [s] |
| `--output` | `augmentation_demo.html` | Plotly HTML の出力先 |

**制約上限**

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--max-lateral-accel` | 3.0 | 横加速度上限 [m/s²](出力ウィンドウ全体) |
| `--max-bridge-speed-gap` | 0.5 | ブリッジ速度乖離上限 [m/s] |
| `--max-bridge-jerk` | 5.0 | ブリッジ縦ジャーク上限 [m/s³] |

**リーク対策(要 `--adaptive-bridge-search`)**

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--randomize-bridge-times` | off | 最小実行可能 M/N の上に乱数で上乗せ |
| `--bridge-time-extra-range` | 1.5 | 上乗せ幅の上限 [s] |
| `--past-bump` | off | 過去履歴にランダムバンプを 1 つ注入 |

**スイープ / ユーティリティ**

| 引数 | 意味 |
|---|---|
| `--sweep` | offset × yaw × N × M × バンプ有無 の全組み合わせを HTML 出力 |
| `--bump-sweep` | 決定論的バンプ(振幅 × 持続 × 開始時刻)のスイープ。各図でバンプなし(Seed)とバンプあり(Best)を比較 |
| `--output-prefix` | スイープ出力のファイル名プレフィックス |
| `--write-pattern-csvs` | 15 パターンの CSV を生成して終了 |
| `--list-patterns` | パターン名を一覧して終了 |

**スイープのグリッド**(`cli.py` の `main()` 内で編集可能):

- `--sweep`: offset = ±1, ±2, ±3 m / yaw = 0, ±5, ±10, ±15 deg /
  N, M = 0.5, 1.0, 1.5, 2.0 s / バンプ = off, on(サフィックス `_bump0` / `_bump1`)
- `--bump-sweep`: 振幅 = −1.2, −0.1, +0.1, +1.2 m / 持続 = 0.8, 1.2, 1.6, 2.0 s /
  開始 = 0.3, 0.8, 1.3, 1.8 s(過去 3 s 窓に収まらない組はスキップ)

### 4.3 Python API の主要エントリポイント

**`run_demo_case(...) -> DemoArtifacts`** — CLI 単一実行と同じフロー

| 引数 | デフォルト | 意味 |
|---|---|---|
| `pattern_name`, `pattern_dir`, `seed` | — | パターンとシード |
| `offset_m` | None → 乱数 | 横オフセット [m] |
| `yaw_offset_deg` | — | 方位オフセット [deg] |
| `recover_time_s` / `past_connect_time_s` | None → 0.1 s | 初期 N / M [s] |
| `max_lateral_accel_mps2` | — | 横加速度上限 |
| `max_bridge_speed_gap_mps` / `max_bridge_jerk_mps3` | 0.5 / 5.0 | ブリッジ制約上限 |
| `adaptive_bridge_search` | True | 実行可能性探索の有効/無効 |
| `randomize_bridge_times` / `bridge_time_extra_range_s` | False / 1.5 | M/N ランダム化 |
| `past_bump` | False | バンプ注入の有効/無効(範囲は内部定数) |

**`augment_trajectory_bidirectional(...) -> BidirectionalAugmentationResult`** — 低レベル API。
`past_lateral_bump: LateralBump | None` で決定論的なバンプを直接指定可能。

**`LateralBump`** — バンプ 1 つの定義(3 フィールド)

| フィールド | 意味 |
|---|---|
| `start_time_s` | t0 から何秒さかのぼった所から始まるか(正の値) |
| `duration_s` | 山の持続時間 [s] |
| `amplitude_m` | 山の高さ [m]。符号で左右 |

**`run_bump_demo_case(...)`** — バンプなしベースライン(Seed)とバンプあり(Best)を
ペアにした `DemoArtifacts` を返す。`--bump-sweep` の 1 ケース分。

### 4.4 内部の安全装置(通常は変更不要)

| パラメータ | 値 | 場所 | 意味 |
|---|---|---|---|
| `max_attempts` | 10 | `randomize_bridge_times_result` | M/N 抽選のリトライ上限 |
| `max_attempts` | 20 | `apply_random_past_history_bump` | バンプ抽選のリトライ上限(全滅時はバンプなし) |
| `min_bump_length_m` | 0.5 | `apply_lateral_bump_to_dense_path` | バンプ区間の弧長がこれ未満(ほぼ停止中)ならスキップ |
| `dense_ds` | 0.05 | 各所 | 稠密パスの弧長刻み [m] |
| `search_step_s` | 0.1 | `search_feasible_result` | M/N 探索のグリッド幅 [s] |

---

## 5. 出力の読み方

### 5.1 サマリー行(CLI 標準出力)

- `Initial past bridge M / future bridge N`: 探索開始値
- `Selected past bridge M / future bridge N`: 最終的に採用された値
- `Past history bump: t in [-a, -b] s, amplitude ±c m`: 適用されたバンプ
- `Lowest passing candidate via <strategy>`: 採用戦略。
  `extend N` / `extend M and N` / `randomize M/N` / `past bump` が `+` で連結される
- 各制約行の `PASS` / `FAIL`: 採用軌跡の制約充足状況
- `End delta at +8s`: 出力ウィンドウ終端での GT との位置差(大オフセット時の
  進捗ラグの指標)

### 5.2 Plotly 図(4 パネル)

| パネル | 内容 |
|---|---|
| Trajectory | GT(灰)、Seed 軌跡(青破線)、Best 軌跡(橙)、オフセット点(赤)、合流点(紫菱形)、0.1 s 間隔の姿勢三角形 |
| Speed | 厳密弧長速度と弦長速度(GT / Seed / Best) |
| Curvature | 曲率(GT / Seed / Best) |
| Lateral Acceleration | 横加速度と上限線、超過点(赤マーカー) |

縦線: `t = 0`(t0)、`t = −M`、`t = +N`(ブリッジ境界)。

---

## 6. 既知の限界と今後の候補

- 過去ブリッジ方式は履歴に離脱の手がかりを残す。M/N ランダム化とバンプで
  緩和しているが、より根本的な対策として **過去履歴の剛体シフトモード**
  (履歴全体を平行にオフセットし、離脱の手がかり自体を消す。自己位置推定誤差の
  模擬として自然)が未実装の候補として残っている。
- バンプは 1 個固定(設計判断)。範囲調整は `core.py` の定数編集で行う。
