# Diffusion Policy học hành động mẫu từ state

`src/Diffusion_Policy_model.py` bổ sung một policy học trực tiếp **lịch sử state → chuỗi action của người chơi**. Model nhận state số, không sử dụng ảnh. Nó có thể điều khiển môi trường Push-T gốc qua `src/run_diffusion_policy.py`, độc lập với World Model và MPC hiện có.

Ý tưởng học phân phối chuỗi hành động bằng khử nhiễu và thực hiện từng đoạn chuỗi dựa trên [Diffusion Policy](https://arxiv.org/abs/2303.04137). Implementation trong project được tự xây dựng bằng PyTorch: mạng residual convolution theo thời gian, không phải bản sao kiến trúc U-Net của repository gốc; không cần cài `diffusers`.

## 1. Vai trò và giới hạn

| Thành phần | Đầu vào | Đầu ra | Cách học/chọn action |
|---|---|---|---|
| World Model hiện có | State và action | State kế tiếp | Học động lực học từ transition |
| MPC hiện có | State, goal, World Model | Chuỗi action được chấm cost | Tìm kiếm CEM lúc chạy; không train riêng |
| Diffusion Policy mới | Lịch sử state | Chuỗi target XY của actuator | Học bắt chước action được ghi trong demonstration |

Policy mới không gọi `CostFunctions.py`, không sử dụng dự đoán của World Model và không được tối ưu trực tiếp theo diện tích overlap, độ giật hay nhiên liệu. Chất lượng di chuyển phụ thuộc vào demonstration và chất lượng học. Đổi `--target-coverage` chỉ đổi điều kiện kết thúc trong runner, không tự thay đổi hành vi đã học.

Target của dữ liệu hiện tại cố định tại `[256, 256, π/4]`. Model chưa nhận goal như một input riêng, do đó chưa hỗ trợ tổng quát hóa có kiểm soát sang vị trí target tùy ý. Một số demonstration gốc dừng ngay sau khi vượt 95% coverage; dữ liệu đó có thể thiếu thao tác căn chỉnh cuối để đạt 99%. Không xem việc bổ sung model hoặc giảm validation loss là bằng chứng đã đạt 99% trong simulator.

## 2. Kế thừa tiền xử lý Zarr

Module phụ trợ [`diffusion_policy_data.py`](../src/diffusion_policy_data.py) gọi lại `preprocess_onestep_data.prepare_data()`; khi dùng cache, gọi `load_data()`. Nó lấy `raw.state`, `raw.action` và metadata `episodes` của cache schema 2. Các nhãn delta-state dùng cho World Model không trở thành nhãn của Diffusion Policy.

- Mặc định tìm đệ quy **tất cả `*.zarr` trong `data/demonstrations/`**, độc lập thư mục hiện tại khi gọi script.
- `--datasets` nhận một hoặc nhiều đường dẫn store, thư mục, hoặc glob. Trùng đường dẫn được loại khi khám phá; dataset sao chép có cùng nội dung được loại bằng SHA-256 trong preprocessing dùng chung.
- Vận tốc được tính bằng sai phân lùi, cùng quy ước của World Model và `StateObserver` lúc chạy.
- Chia train/validation theo nhóm pose khởi tạo của cả episode, kể cả khi các episode nằm trong nhiều dataset. Không chia ngẫu nhiên từng timestamp hoặc từng window.
- Episode boundaries, source, signature và split được giữ trong provenance của checkpoint/history. Zarr nguồn chỉ được đọc; ảnh không được đưa vào model.
- Cần ít nhất hai nhóm pose khởi tạo có transition để preprocessing dùng chung tạo được cả train và validation.

State vật lý là:

$$
s_t=[p^a_x,p^a_y,v^a_x,v^a_y,p^T_x,p^T_y,v^T_x,v^T_y,\theta^T,\omega^T]\in\mathbb{R}^{10}.
$$

Vận tốc tuyến tính dùng $(p_t-p_{t-1})/\Delta t$; vận tốc góc dùng chênh lệch góc sau unwrap. Vận tốc đầu episode bằng 0. Control mặc định giữ **10 Hz**, $\Delta t=0.1$ s; physics trong môi trường gốc vẫn chạy 100 Hz.

`encode_states()` thay góc bằng `sin(theta), cos(theta)` để tránh bước nhảy của biểu diễn góc tại ±π. Thứ tự 11 feature là `agent XY, agent VXY, T XY, T VXY, sin(theta), cos(theta), omega`. Mean và sample standard deviation được tính trên các state thuộc **train**, mỗi state được tính một lần, kể cả state được ghi cuối episode. Standard deviation có sàn `1e-6`.

Action là **vị trí target XY tuyệt đối của actuator** tại timestamp hiện tại, không phải vận tốc, lực hoặc delta vị trí. Chuẩn hóa dùng miền workspace đã biết:

$$
\tilde a = 2\frac{a-0}{512-0}-1=\frac a{256}-1.
$$

Dữ liệu đang có một số target bằng `513` do recorder chuột. Nhãn này được giữ nguyên, tương ứng `1.00390625` sau chuẩn hóa; không sửa Zarr và không âm thầm cắt nhãn train. Bộ đọc cho phép sai lệch biên tối đa một đơn vị (`[-1,513]`), từ chối vượt xa hơn. Normalizer lưu min/max action và số mẫu ngoài `[0,512]` để kiểm tra; các thống kê này không điều chỉnh scale. Khi xuất action cho simulator, policy luôn clip về `[0,512]`.

## 3. Căn timestamp và tạo chuỗi

Ký hiệu `O = obs_horizon`, `H = prediction_horizon`, `A = action_horizon`. Với điểm bắt đầu action hiện tại `t`, một mẫu train gồm:

$$
c_t=[s_{t-O+1},\ldots,s_t],\qquad X_0=[a_t,a_{t+1},\ldots,a_{t+H-1}].
$$

State cuối của lịch sử là state **trước** khi áp dụng action đầu tiên `a_t`, đúng cách recorder gốc ghi dữ liệu. Không lấy state tương lai để điều kiện hóa policy. Ở đầu episode, lịch sử thiếu được điền bằng state đầu tiên. Ở cuối episode, chuỗi action được điền bằng action cuối cho đủ kích thước, nhưng các phần tử thêm được đánh dấu `mask=False` và không đóng góp vào loss.

Ví dụ episode có `s0,s1,s2` và `a0,a1,a2`, với `O=2,H=4`: mẫu `t=0` dùng `[s0,s0]`, target `[a0,a1,a2,a2]`, mask `[1,1,1,0]`; mẫu `t=2` dùng `[s1,s2]`, target `[a2,a2,a2,a2]`, mask `[1,0,0,0]`. Không lấy phần đầu của episode tiếp theo để nối chuỗi.

**Action cuối được ghi trong mỗi episode vẫn là nhãn hợp lệ.** Khác với dự đoán next-state, imitation không cần state sau action đó để tạo target. Vì vậy số mẫu policy là số timestamp được ghi, lớn hơn số transition của World Model. Một action có thể xuất hiện trong nhiều future window; đây là cách huấn luyện chuỗi trượt, không phải các mẫu độc lập theo thời gian.

## 4. Công thức diffusion và denoiser

### 4.1 Forward diffusion và loss

Với chuỗi action đã chuẩn hóa $X_0$, lấy timestep nhiễu $k$ ngẫu nhiên và $\epsilon\sim\mathcal N(0,I)$:

$$
X_k=\sqrt{\bar\alpha_k}X_0+\sqrt{1-\bar\alpha_k}\epsilon.
$$

Network nhận $(X_k,k,c_t)$ và dự đoán $\hat\epsilon=\epsilon_\theta(X_k,k,c_t)$. Mục tiêu là epsilon MSE theo cách học khử nhiễu của [DDPM](https://arxiv.org/abs/2006.11239), với mask riêng cho padding:

$$
\mathcal L=\frac{\sum_{b,h,d}m_{b,h}(\hat\epsilon_{b,h,d}-\epsilon_{b,h,d})^2}{2\sum_{b,h}m_{b,h}},\qquad d\in\{x,y\}.
$$

`masked_noise_loss()` loại phần tail padding khỏi tử số và mẫu số. Mỗi epoch báo trung bình có trọng số theo số thành phần action thật, không lấy trung bình các batch với trọng số bằng nhau. Validation loss có đơn vị noise MSE, không phải sai số pixel của action hay tỷ lệ thành công.

### 4.2 Cosine schedule

`DiffusionSchedule` dùng cosine schedule theo [Improved DDPM](https://arxiv.org/abs/2102.09672). Với tổng số mức nhiễu `T` và $s=0.008$:

$$
f(u)=\cos^2\left(\frac{u/T+s}{1+s}\frac\pi2\right),\quad
\beta_k=\operatorname{clip}\left(1-\frac{f(k)}{f(k-1)},10^{-8},0.999\right),\quad
\bar\alpha_k=\prod_{i=1}^{k}(1-\beta_i).
$$

Ở công thức trên $k=1,\ldots,T$; code lưu các mức này ở chỉ số `0..T-1`. Chỉ số `-1` ở reverse sampler biểu diễn chuỗi sạch với $\bar\alpha=1$, không phải indexing phần tử cuối của tensor.

### 4.3 Kiến trúc mạng

`ConditionalTemporalDenoiser` nhận noisy actions `(B,H,2)` và lịch sử đã chuẩn hóa `(B,O,11)`. Timestep được mã hóa sinusoidal, đưa qua MLP, rồi ghép với toàn bộ lịch sử quan sát để tạo conditioning. Action đi qua convolution `1×1`, một dãy residual block temporal và convolution đầu ra thành epsilon `(B,H,2)`.

Mỗi block gồm hai convolution kernel 3, GroupNorm và SiLU. Convolution đầu sử dụng dilation lặp `1,2,4,8`; conditioning dự đoán scale/shift theo kênh:

$$
\operatorname{FiLM}(h,c)=(1+\gamma(c))h+\beta(c),\qquad y=(x+h')/\sqrt2.
$$

Convolution nhìn các vị trí trong **chuỗi action đang được khử nhiễu**, kể cả các vị trí sau `t`; điều này không cung cấp state tương lai. Toàn bộ điều kiện quan sát vẫn kết thúc ở `s_t`.

### 4.4 DDIM lúc suy luận

Sampler bắt đầu bằng nhiễu Gaussian `(1,H,2)`, chọn `inference_steps` mốc từ `T-1` về `0`, rồi dùng [DDIM](https://arxiv.org/abs/2010.02502) với `eta=0`. Với mốc đang xét $k$ và mốc được chọn tiếp theo $j$:

$$
\hat X_0=\operatorname{clip}\left(\frac{X_k-\sqrt{1-\bar\alpha_k}\hat\epsilon}{\sqrt{\bar\alpha_k}},-1,1\right),\qquad
\epsilon'=\frac{X_k-\sqrt{\bar\alpha_k}\hat X_0}{\sqrt{1-\bar\alpha_k}},
$$

$$
X_j=\sqrt{\bar\alpha_j}\hat X_0+\sqrt{1-\bar\alpha_j}\epsilon'.
$$

Sau clip, code tính lại noise direction để nhất quán với $\hat X_0$. Bước cuối dùng $\bar\alpha_j=1$. Không thêm nhiễu mới ở mỗi bước reverse; chuỗi vẫn phụ thuộc nhiễu khởi tạo. Cùng checkpoint, input, seed và môi trường tính toán cho phép lặp lại kết quả; không cam kết bitwise giống nhau giữa CPU/GPU hoặc phiên bản thư viện.

## 5. Mặc định, training và checkpoint

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `obs_horizon` | 2 | Hai state liên tiếp; khoảng cách giữa chúng là 0,1 s |
| `prediction_horizon` | 16 | Sinh 16 action, tương ứng 1,6 s lệnh điều khiển |
| `action_horizon` | 8 | Thực hiện 8 action rồi sinh chuỗi mới |
| `diffusion_steps` / `inference_steps` | 100 / 20 | Mức nhiễu khi train / số bước denoise khi chạy |
| `channels` / `residual_blocks` / `time_embed_dim` | 64 / 4 / 64 | Kích thước mạng mặc định |
| Epoch / batch size / learning rate | 100 / 256 / `1e-4` | Thiết lập train ban đầu |
| Weight decay / gradient clipping | `1e-6` / 1 | AdamW / giới hạn gradient norm |

Sau mỗi optimizer step `u`, EMA cập nhật trọng số theo $\theta_{EMA}\leftarrow d_u\theta_{EMA}+(1-d_u)\theta$, với $d_u=\min(0.995,(1+u)/(10+u))$. Validation dùng trọng số EMA. Noise và timestep của validation được tái tạo bằng seed cố định mỗi epoch để so sánh các epoch trên cùng phép thử; train dùng chuỗi noise tiếp tục thay đổi.

`*_best.pt` lưu EMA tại epoch có validation epsilon MSE nhỏ nhất. `*_last.pt` lưu EMA của epoch cuối; JSON history được cập nhật từng epoch. Checkpoint schema 1 chứa architecture `conditional_temporal_diffusion_v1`, config, normalizer, weights CPU và metadata nguồn/split. Ghi file thông qua file tạm rồi thay thế. `load_policy()` dùng `torch.load(..., weights_only=True)`; checkpoint không chứa optimizer state hoặc toàn bộ trạng thái RNG để resume chính xác.

`--resume` là **fine-tune** từ EMA checkpoint: giữ config và normalization cũ, tạo AdamW mới, bắt đầu đếm epoch và EMA updates lại. Code từ chối khi nhóm pose đã có trong checkpoint bị đổi từ train sang validation hoặc ngược lại. Khi bổ sung dataset mới hoặc đổi seed, việc chia lại nhóm pose có thể làm phát sinh lỗi này; dùng cache giữ đúng assignment trước đó, hoặc train model mới từ đầu. Không coi episode từng dùng để train là dữ liệu đánh giá độc lập.

## 6. Lệnh sử dụng

Chạy từ thư mục `pusht-research/`, sau `conda activate pusht-sim` hoặc môi trường project đã cài. Train từ toàn bộ Zarr trong `data/demonstrations/`:

```bash
python -B src/Diffusion_Policy_model.py \
  --epochs 100 --batch-size 256 --lr 1e-4 \
  --seed 42 --device cpu --torch-threads 1 \
  --output checkpoints/DiffusionPolicy_best.pt \
  --last-output checkpoints/DiffusionPolicy_last.pt \
  --history experiments/diffusion_policy_training.json
```

Chọn một thư mục khác, danh sách store hoặc glob được mở rộng bởi chương trình:

```bash
python -B src/Diffusion_Policy_model.py --datasets data/my_demos --epochs 100
python -B src/Diffusion_Policy_model.py \
  --datasets data/demonstrations/pusht_demo.zarr data/demonstrations/demo_actions_v1.zarr
python -B src/Diffusion_Policy_model.py --datasets 'data/**/*.zarr'
```

Dùng lại cache schema 2 đã tạo bởi preprocessing; không kết hợp `--data-pkl` với `--datasets`. Split trong cache được giữ nguyên, còn `--control-hz` vẫn phải khớp cache:

```bash
python -B src/Diffusion_Policy_model.py --data-pkl data/processed/demo_actions_v1.pkl
python -B src/Diffusion_Policy_model.py \
  --resume checkpoints/DiffusionPolicy_best.pt --epochs 30 --lr 3e-5 \
  --output checkpoints/DiffusionPolicy_finetuned_best.pt \
  --last-output checkpoints/DiffusionPolicy_finetuned_last.pt \
  --history experiments/diffusion_policy_finetuned.json
```

`--device cuda` dùng được khi môi trường PyTorch/CUDA phù hợp. CPU là mặc định và không cần đổi dependency của project. Có thể chỉnh `--channels`, `--residual-blocks`, `--obs-horizon`, `--prediction-horizon` trước lần train mới; không thay kiến trúc khi fine-tune checkpoint đã có.

Chạy liên tục với simulator gốc, biểu đồ và ngưỡng qua màn 99%:

```bash
python -B src/run_diffusion_policy.py \
  --checkpoint checkpoints/DiffusionPolicy_best.pt \
  --continuous --start-seed 1000 --max-steps 300 \
  --target-coverage 0.99 --render --realtime-plotting --pace \
  --output results/diffusion_policy_live
```

Chạy hữu hạn để đánh giá, không mở GUI:

```bash
python -B src/run_diffusion_policy.py \
  --checkpoint checkpoints/DiffusionPolicy_best.pt \
  --seeds 1000 1001 1002 --max-steps 300 \
  --output results/diffusion_policy_evaluation
```

Runner gọi `policy.act(state)` tại **mọi bước** để cập nhật lịch sử, kể cả lúc đang lấy action từ queue. Sau mỗi `A` action, policy dùng lịch sử mới nhất để sample chuỗi mới. Có thể thêm `--action-horizon 1` để phản hồi mỗi bước, hoặc giảm `--inference-steps` để giảm số lần gọi denoiser. Thay đổi này có thể ảnh hưởng chất lượng và chi phí tính toán; cần đánh giá lại. `--pace` cố giữ 10 Hz khi thời gian tính toán cho phép.

Coverage được đo bằng hình học simulator. Đạt `>= target_coverage` thì lưu và chuyển episode; hết `max_steps` thì lưu nhãn `time_limit` và chuyển episode, không tính thành công. Runner tiếp tục sau flag done gốc >95% nếu chưa đạt ngưỡng yêu cầu. Q/Esc/đóng cửa sổ simulation hoặc biểu đồ lưu lượt đang dở và dừng; Ctrl+C cũng được xử lý. Space trong biểu đồ chỉ tạm dừng hiển thị.

Mỗi episode có `episode_SEED.npz`, `.json` và `.png` nếu không dùng `--no-plots`; `summary.json` cập nhật sau từng episode. Lượt bị người dùng ngắt không được tính vào success rate. NPZ lưu state/action thực, coverage, phần diện tích chưa phủ, contact, inference/sampling time, các action chunk đã sample và timestamp bắt đầu chunk. File không chứa dự đoán next-state hoặc cost MPC. Đây là trace đánh giá, chưa phải Zarr demonstration dùng để train trực tiếp.

## 7. Interface chính

| Interface | Đầu vào và kết quả |
|---|---|
| `discover_zarr_datasets(inputs=None)` | Đường dẫn/thư mục/glob → danh sách store đã resolve, loại trùng và sắp xếp |
| `prepare_policy_data(datasets=None, ..., data_pkl=None)` | Tạo hoặc nạp cache dùng chung schema 2, kiểm tra raw arrays, split và control period |
| `make_policy_normalizer(cache)` | Dict mean/std 11D train-only, scale action cố định, dt và thống kê biên |
| `PolicySequenceDataset(cache, split, O, H, normalizer)` | Mỗi item: `obs (O,11)`, `action (H,2)`, `mask (H,)`; `indices` là `(episode_index, local_step)` |
| `train_policy(datasets=None, *, ...)` | Train, lưu best/last/history → `(best_policy, history)` |
| `load_policy(path, device="cpu")` | Nạp checkpoint → `DiffusionPolicy` ở chế độ inference |
| `save_policy(policy, path, metadata=None)` | Lưu checkpoint portable; không lưu history queue đang chạy |
| `policy.predict_action(history, seed=None, inference_steps=None)` | Raw state `(T,10)` hoặc `(10,)` → action vật lý `(H,2)`; lấy O state gần nhất và pad nếu thiếu; không đổi execution queue |
| `policy.reset(seed=None)` | Xóa observation history/action queue, tạo RNG cho episode mới |
| `policy.act(state)` | Raw state `(10,)` → action `(2,)`; cập nhật history, sample khi queue hết |
| `run_episode(env, policy, seed, ...)` | Trả `(metrics, arrays, policy_diagnostics)`; hỗ trợ plotting và ngắt có lưu trace |

`last_replanned`, `last_plan`, `last_sampling_seconds` cho biết bước hiện tại có sample lại hay không, chuỗi được sample gần nhất và thời gian sampling. `policy.dt` là control period được lưu trong normalization. Runner từ chối checkpoint có control period khác môi trường.

## 8. Kiểm chứng và diễn giải kết quả

```bash
python -B -m unittest discover -s tests -p 'test_diffusion*.py' -v
python -B -m unittest discover -s tests -v
```

Kiểm thử tập trung vào alignment state/action, tính nhân quả, ranh giới episode, padding mask, normalization train-only, chia nhóm pose, checkpoint và diffusion sampling; runtime kiểm tra ngưỡng 99%, xử lý ngắt, lưu nhiều episode và chạy môi trường gốc. Test chức năng không đo năng lực hoàn thành nhiệm vụ của checkpoint sau train.

Kiểm chứng thực tế ngày 15/11/2026:

- **99 test toàn project pass**, gồm 27 test mới: 9 dữ liệu, 9 model, 9 runtime. Có test oracle kiểm tra công thức DDIM, training trên Zarr nhỏ và regression chạy CLI với môi trường gốc.
- Train thử cấu hình mặc định **1 epoch** trên `demo_actions_v1.zarr` và `pusht_demo.zarr`: 224 episode, 41.582 mẫu; train 36.093, validation 5.489. Train epsilon MSE `0.292397`, validation EMA epsilon MSE `0.146142`; epoch khoảng 7,2 s trên CPU của workspace với một Torch thread.
- Có 3.945 action chứa tọa độ 513, được giữ trong nhãn train đúng quy ước đã nêu.
- Checkpoint thử `results/diffusion_smoke/policy_best.pt` đã được nạp lại và chạy simulator gốc seed 1000 trong 24 bước, sinh 3 action chunk. NPZ lưu 25 state và 24 action; JSON, PNG và summary đã tạo. Sampling trung bình khoảng 11,1 ms/chunk trong lần chạy ngắn này, không phải bảo đảm realtime trên mọi máy.
- Lượt thử kết thúc `time_limit`, max coverage khoảng 17,84%; đây là smoke test với model mới train một epoch, **không phải benchmark chất lượng sau train đầy đủ**. Các artifact thử nằm riêng tại `results/diffusion_smoke/`, không thay thế `checkpoints/DiffusionPolicy_best.pt` của lệnh train chính thức.

Đánh giá model bằng cả validation epsilon MSE và rollout thực với seed chưa dùng để chọn cấu hình. Ghi rõ checkpoint, số epoch, seed, `action_horizon`, số inference step, `max_steps` và ngưỡng coverage khi báo cáo success rate. Thời gian mô phỏng được tính bằng số action hoàn thành nhân `dt`; wall time và độ trễ inference được ghi riêng. Không suy ra khả năng realtime hoặc mức độ an toàn cơ khí chỉ từ hành động nhìn có vẻ mượt.

## 9. Chạy checkpoint pretrained gốc trong cùng giao diện

Cập nhật ngày 16/11/2026: `run_pretrained_diffusion.py` dùng vòng lặp của `run_diffusion_policy.py`, nhưng tải model gốc qua `pretrained_diffusion_policy.py`. Cửa sổ simulator, biểu đồ, continuous, Q/Esc và cách lưu NPZ/JSON/PNG được dùng chung. Mặc định **MPC và pretrained là coverage ≥95%**; policy tự viết vẫn mặc định 99%. Khi so sánh cả ba, truyền rõ `--target-coverage 0.95` cho mọi runner.

Adapter chỉ hỗ trợ checkpoint `DiffusionUnetLowdimPolicy` của Push-T với keypoint đầy đủ: 9 điểm của T và XY actuator, không có latency/past-action conditioning. Model image hoặc Transformer được từ chối để tránh đưa sai observation vào checkpoint. Mặc định tải trọng số EMA, normalizer và DDPM scheduler từ checkpoint, giữ 100 bước denoise. `--sampler ddim` cho phép đổi sang DDIM để lấy mẫu ít bước với cùng trọng số; đây là một cấu hình đánh giá khác, cần đo lại chất lượng. `map_location="cpu"` cho phép đọc checkpoint trước khi chọn CPU/GPU. Các dependency gốc được import khi gọi loader; model tự viết không cần cài chúng.

Với điểm local $q_i$ của T, feature đầu vào được dựng bằng $k_i=R(\theta_T)q_i+p_T$, rồi nối `[k_1.x,k_1.y,...,k_9.x,k_9.y,agent_x,agent_y]`. Map local lấy trực tiếp từ `PushTKeypointsEnv.genenerate_keypoint_manager_params()` của source gốc. Các vận tốc trong state 10D phục vụ biểu đồ/metric, không được ghép vào observation 20D của model này.

Checkpoint có `oa_step_convention=True`, `O=2`, `H=16`: action đầu tiên cần thực thi nằm tại **index `O-1=1`** của `action_pred`. Adapter bỏ index 0 thuộc quá khứ, giữ 15 action hiện tại/tương lai, kiểm tra 8 action đầu khớp output `action` của policy gốc rồi thực thi theo queue. Không lấy action quá khứ làm lệnh hiện tại. Mỗi state mới đều cập nhật history, kể cả khi queue chưa hết; thiếu history đầu lượt được pad bằng state đầu.

Policy gốc dùng RNG Torch toàn cục trong sampler. Adapter dùng `torch.random.fork_rng` để lưu/phục hồi trạng thái RNG của caller và duy trì chuỗi RNG riêng cho từng episode. Reset xóa history/queue và khởi tạo lại theo seed. Action được clip về `[0,512]` cho cùng giới hạn runtime; điều này có thể khác native evaluator ở các target vượt biên nhẹ.

| API | Chức năng |
|---|---|
| `state_to_keypoints(states, block_keypoints)` | State `(...,10)` → observation `(...,20)`, transform theo body origin thực |
| `load_policy(path, device="cpu")` trong adapter | Nạp checkpoint gốc, kiểm tra schema/config, lấy EMA nếu được bật |
| `PretrainedDiffusionPolicy.predict_action(history, seed=None, inference_steps=None)` | Lấy mẫu chuỗi hiện tại/tương lai `(15,2)` với checkpoint đã chọn |
| `PretrainedDiffusionPolicy.act(state)` | Cập nhật state mỗi bước, sample lại khi queue hết, trả action `(2,)` |
| `PretrainedPolicyConfig` | Lưu native horizon, offset, future horizon, số action thực thi, sampler và denoising steps; kiểm tra override hợp lệ |
| `make_ddim_scheduler(original_schedule)` | Tạo DDIM từ noise schedule của checkpoint, import Diffusers khi cần và cache trên adapter |
| `run_pretrained_diffusion.main(argv=None)` | Dùng runner chung, default checkpoint gốc và threshold 0,95 |

`summary.json` ghi config thật của checkpoint trong `policy_metadata`, hash checkpoint, sampler và số bước trong `policy_config`, cùng `environment_legacy=False`. Native `eval.py` là phép đánh giá riêng dùng preset của checkpoint (`legacy=True`, done >95%, score reward đã chuẩn hóa), trong khi runner live dùng chung môi trường và predicate ≥95% với MPC. Không gọi mean score native là success rate của runner mới.

Chạy trong môi trường `robodiff` đã chuẩn bị:

```bash
cd ~/AI-application/pusht-research
conda activate robodiff
python -B src/run_pretrained_diffusion.py \
  --continuous --start-seed 1000 --max-steps 300 \
  --target-coverage 0.95 --device cpu \
  --render --realtime-plotting --pace \
  --output results/github_diffusion_live
```

Vòng lặp lấy mẫu đồng bộ: chạy hết queue 8 action mới gọi model tạo chuỗi tiếp theo. Episode CPU `results/github_live_95/episode_1000.json` ghi 132 action và 17 lần lấy mẫu, trung bình 2,293 giây/lần; lấy action từ queue trung bình chỉ 0,049 ms. Tổng inference khoảng 38,98 giây trong 54,85 giây thực, trong khi thời gian mô phỏng là 13,2 giây. Đây là nguyên nhân chính của các lần GUI khựng định kỳ; chưa có phép đo tách riêng chi phí biểu đồ, mô phỏng và pacing. Giảm `--action-horizon` xuống 1 khiến chờ inference thường xuyên hơn. Bỏ `--realtime-plotting` có thể giảm chi phí hiển thị, nhưng không loại bỏ việc chờ model.

Để thử DDIM 20 bước, nhấn **Q** lưu và kết thúc phiên cũ, sau đó chạy phiên riêng:

```bash
conda activate robodiff
cd ~/AI-application/pusht-research
python -B src/run_pretrained_diffusion.py \
  --checkpoint checkpoints/pretrained/PushT_original_lowdim.ckpt \
  --sampler ddim --inference-steps 20 \
  --continuous --start-seed 1000 --max-steps 300 \
  --target-coverage 0.95 --device cpu --threads 1 \
  --render --realtime-plotting --pace \
  --output results/github_live_ddim20_95
```

Giữ baseline bằng `--sampler ddpm --inference-steps 100` hoặc để mặc định. Không dùng `--inference-steps 20` riêng: DDPM trong Diffusers 0.11.1 lấy hệ số ở `t-1` ngay cả khi lịch lấy mẫu bỏ qua timestep, nên runner từ chối giảm bước DDPM. DDIM xử lý lịch ít bước bằng scheduler tương ứng; checkpoint không đổi nhưng kết quả có thể đổi, phải báo cáo sampler/số bước và đánh giá lại success rate. Ít bước không bảo đảm realtime; CUDA của `robodiff` hiện trả về `False`, chỉ chọn `--device cuda:0` sau khi `torch.cuda.is_available()` trả về `True`.

Adapter tạo `DDIMScheduler.from_config(original_schedule.config)` để giữ beta schedule và kiểu dự đoán đã train. Với `eta=0`, bước chuyển DDIM là xác định khi biết noise khởi tạo và observation; noise khởi tạo vẫn phụ thuộc seed. Scheduler, `num_inference_steps` và `kwargs` của native policy được đổi tạm thời trong `predict_action`, rồi khôi phục bằng `finally` kể cả khi inference lỗi. Nguồn đối chiếu là [DDPM v0.11.1](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py) và [DDIM v0.11.1](https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddim.py).

Đo latency với checkpoint thật ngày 16/11/2026: cùng observation đầu episode 1000, CPU/1 thread, 3 lần lấy mẫu mỗi sampler; DDPM100 trung bình **2,278 giây**, DDIM20 **0,446 giây**, tỷ số **5,11 lần**. Forward hook xác nhận đúng 100 và 20 lần gọi denoiser. Artifact `results/pretrained_latency_benchmark/summary.json` ghi từng lần đo và phạm vi phép đo; chưa suy ra success rate hoặc bảo đảm realtime từ số đo này.

Đã kiểm tra **119 test toàn project** trong 5,892 giây, trong đó 11 test adapter và 9 test tích hợp pretrained; phép đổi keypoint được so với observation của environment gốc, kiểm tra timestamp action, RNG, queue, ngưỡng 95% chung với MPC và cấu hình DDPM/DDIM. Hai file test pretrained cũng pass trong môi trường `robodiff`. Checkpoint thật đã chạy headless và mở cả hai cửa sổ; Q ở từng cửa sổ đều lưu đúng các lượt trước và lượt dở. Lượt kiểm tra DDIM20 mới chạy 16 bước điều khiển, lưu 2 chuỗi action tại bước 0 và 8, trung bình 0,456 giây/lần lấy mẫu; artifact ở `results/pretrained_ddim20_smoke/`. Lượt ngắn này dừng do giới hạn 16 bước, không dùng làm phép đo thành công. Chưa có benchmark đầy đủ ba controller từ các smoke test này.
