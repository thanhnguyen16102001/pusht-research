# Push-T — World Model, MPC và Diffusion Policy

Điều khiển actuator đẩy khối T vào target bằng dữ liệu tự thu thập, chạy trên môi trường [Push-T gốc](https://github.com/real-stanford/diffusion_policy). Control **10 Hz**, mô phỏng vật lý **100 Hz**.

| Controller | Model cần chuẩn bị | Môi trường Conda |
|---|---|---|
| MPC local | Train World Model; **MPC không train riêng** | `pusht-sim` hoặc `pusht-research` |
| Diffusion Policy tự viết | Train policy từ toàn bộ `.zarr` | `pusht-sim` hoặc `pusht-research` |
| Diffusion Policy GitHub | Tải checkpoint pretrained của tác giả | `robodiff` |

## 1. Chuẩn bị

Hai thư mục `pusht-research/` và `diffusion_policy/` nằm cạnh nhau trong `~/AI-application/`. Nếu đặt workspace ở nơi khác, thay đường dẫn trong các lệnh.

Repository bàn giao gồm source, dữ liệu, checkpoint và kết quả. Checkpoint pretrained lớn dùng **Git LFS**: cài Git LFS trước khi clone; nếu đã clone, chạy `git lfs install` rồi `git lfs pull` trong repository để tải trọng số đầy đủ.

```bash
cd ~/AI-application/pusht-research
```

Nếu chưa có source mô phỏng:

```bash
git clone https://github.com/real-stanford/diffusion_policy.git ../diffusion_policy
git -C ../diffusion_policy checkout 5ba07ac6661db573af695b419a7947ecb704690f
```

Dùng `conda activate pusht-sim` nếu đã cài môi trường local. Nếu cài mới:

```bash
conda env create -f environment_research.yaml
conda activate pusht-research
```

Các bước thu dữ liệu và train bên dưới chạy từ `pusht-research/`, trong môi trường local này. Cửa sổ simulation cần desktop/X11/WSLg.

## 2. Thu và xem dữ liệu

```bash
mkdir -p data/demonstrations
python -B ../diffusion_policy/demo_pusht.py -o data/demonstrations/my_demo.zarr

python -B src/episode_kinematics.py --visualize \
  --dataset data/demonstrations/my_demo.zarr --episode 0
```

Điều khiển bằng chuột; **Q** thoát, **R** thử lại, **Space** tạm dừng. Recorder gốc lưu lượt thành công, không lưu lượt đang dở khi thoát. Snapshot bàn giao đã kèm hai dataset trong `data/demonstrations/`; lệnh trên dùng để thu thêm dữ liệu.

## 3. Train model

### World Model cho MPC

```bash
python -B src/Model_MLP_OneStepPredict.py \
  --dataset data/demonstrations/pusht_demo.zarr data/demonstrations/demo_actions_v1.zarr \
  --architecture contact_local_v2 --epochs 200 --val-ratio 0.1 \
  --seed 42 --device cpu --torch-threads 1 \
  --output checkpoints/WorldModel_contact_best.pt \
  --last-output checkpoints/WorldModel_contact_last.pt \
  --history experiments/world_model_contact_training.json
```

Thay danh sách `--dataset` bằng dữ liệu của bạn, ví dụ `data/demonstrations/my_demo.zarr`. Cần nhiều episode thuộc ít nhất hai nhóm pose khởi tạo để chia train/validation. World Model học **state + action → state kế tiếp**; MPC tối ưu sai lệch T, diện tích target chưa phủ, biến thiên vận tốc, quãng đường và thời gian. Đổi cost hoặc ngưỡng coverage không yêu cầu train lại MPC/World Model.

### Diffusion Policy tự viết

Lệnh này tìm tất cả `.zarr` trong `data/demonstrations/` và thư mục con, dùng lại preprocessing của World Model:

```bash
python -B src/Diffusion_Policy_model.py \
  --epochs 100 --batch-size 256 --device cpu --torch-threads 1 \
  --output checkpoints/DiffusionPolicy_best.pt \
  --last-output checkpoints/DiffusionPolicy_last.pt \
  --history experiments/diffusion_policy_training.json
```

Chọn nguồn khác bằng `--datasets /duong/dan/du_lieu`. Policy học bắt chước action, không cần World Model. Dùng checkpoint `*_best.pt` để chạy; hướng dẫn fine-tune và cấu hình nằm trong [tài liệu policy](docs/diffusion_policy.md).

## 4. Chuẩn bị model GitHub

Nếu đã có `robodiff` và checkpoint, chuyển đến **bước 6**. Nếu chưa có môi trường:

```bash
cd ~/AI-application/diffusion_policy
conda env create -f conda_environment.yaml
conda activate robodiff
```

Nếu phần Conda đã báo `Executing transaction: done` nhưng pip lỗi `MUJOCO_PATH`, có thể hoàn tất dependency cần cho Push-T bằng lệnh dưới. Pin Hub cũng xử lý lỗi `cached_download` với Diffusers cũ:

```bash
conda activate robodiff
python -m pip install --only-binary=:all: "pygame==2.1.2" "huggingface-hub==0.25.2"
```

Tải checkpoint chính thức khoảng 1 GiB, chỉ cần thực hiện một lần:

```bash
cd ~/AI-application/pusht-research
mkdir -p checkpoints/pretrained
curl --fail --location --continue-at - \
  --output checkpoints/pretrained/PushT_original_lowdim.ckpt \
  'https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt'
```

## 5. Sử dụng, so sánh và tài liệu

- **Q/Esc hoặc đóng cửa sổ**: lưu episode đang chạy và thoát. Các episode trước được giữ.
- Coverage **≥95%** trong các lệnh bên dưới: lưu thành công rồi tự qua lượt. Hết 300 bước: lưu `time_limit` rồi qua lượt.
- `--render` mở simulator, `--realtime-plotting` mở biểu đồ, `--pace` giữ nhịp 10 Hz khi tính toán cho phép. Space trong biểu đồ chỉ tạm dừng hiển thị.
- Mỗi lượt lưu NPZ/JSON/PNG; `summary.json` tổng hợp coverage, tỷ lệ thành công, thời gian, quãng đường và độ mượt. Output phải là thư mục mới; hai lệnh cuối tự gắn thời gian vào tên thư mục.

Chạy **Diffusion Policy tự viết** sau khi train, trong môi trường local:

```bash
conda activate pusht-sim
cd ~/AI-application/pusht-research
python -B src/run_diffusion_policy.py \
  --checkpoint checkpoints/DiffusionPolicy_best.pt \
  --continuous --start-seed 1000 --max-steps 300 --target-coverage 0.95 \
  --render --realtime-plotting --pace \
  --output "results/own_diffusion_$(date +%Y%m%d_%H%M%S)"
```

Nếu cài bằng manifest mới, thay `conda activate pusht-sim` bằng `conda activate pusht-research` trong các lệnh local. Policy tự viết mặc định coverage 99%; lệnh trên đặt 95% để so sánh với MPC và GitHub.

Đánh giá hữu hạn: thay `--continuous --start-seed 1000` bằng `--seeds 1000 1001 1002 1003 1004`, bỏ ba flag hiển thị/pace nếu chạy headless. Giữ cùng seed, ngưỡng và số bước tối đa; so sánh các `summary.json`, đồng thời ghi rõ checkpoint, sampler và thiết bị. `eval.py` của repo gốc chạy batch và xuất video, không mở giao diện live; score native khác success rate của các runner này.

Kiểm thử trong môi trường local: `python -B -m unittest discover -s tests -v`. Mốc gần nhất: **119 tests pass**; đây không phải số đo chất lượng điều khiển.

Tài liệu chi tiết: **[ngữ cảnh bàn giao cho AI viết báo cáo 15–20 trang](overall_context_detail.md)**, [báo cáo kỹ thuật](Detail_Report.md), [Diffusion Policy và chế độ DDIM](docs/diffusion_policy.md), [cost diện tích chưa phủ](docs/overlap_refinement.md), [nguồn và giấy phép thành phần gốc](THIRD_PARTY_NOTICES.md).

## 6. Chạy model GitHub trong simulator

Dùng **DDIM 20 bước** để giảm khựng trên CPU, không cần train lại. Phép đo hiện tại giảm thời gian lấy mẫu từ khoảng **2,28 s xuống 0,45 s**; chất lượng cần đánh giá lại khi đổi sampler. Để dùng cấu hình gốc, thay cặp flag bằng `--sampler ddpm --inference-steps 100` và lưu kết quả riêng. Không giảm bước DDPM bằng riêng `--inference-steps 20`.

```bash
conda activate robodiff
cd ~/AI-application/pusht-research

python -B src/run_pretrained_diffusion.py \
  --checkpoint checkpoints/pretrained/PushT_original_lowdim.ckpt \
  --sampler ddim --inference-steps 20 \
  --continuous --start-seed 1000 --max-steps 300 \
  --target-coverage 0.95 --device cpu --threads 1 \
  --render --realtime-plotting --pace \
  --output "results/github_ddim20_$(date +%Y%m%d_%H%M%S)"
```

## 7. Chạy MPC local trong simulator

Dùng World Model đã train và cost overlap. Nhấn **Q** để lưu, thoát phiên đang chạy trước khi chuyển controller. Nếu cài mới bằng manifest, dùng `conda activate pusht-research` thay cho `pusht-sim`.

```bash
conda activate pusht-sim
cd ~/AI-application/pusht-research

python -B src/run_mpc.py \
  --checkpoint checkpoints/WorldModel_contact_best.pt \
  --config configs/mpc_overlap.json \
  --continuous --start-seed 1000 --max-steps 300 \
  --target-coverage 0.95 --device cpu --threads 1 \
  --render --realtime-plotting --pace \
  --output "results/mpc_95_$(date +%Y%m%d_%H%M%S)"
```
