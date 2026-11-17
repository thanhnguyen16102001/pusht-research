# Nguồn tham khảo và thành phần của bên thứ ba

Dự án này sử dụng môi trường Push-T và recorder từ [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), commit tham chiếu `5ba07ac6661db573af695b419a7947ecb704690f`. Source gốc được clone ở thư mục bên cạnh, không gộp vào source tự xây dựng trong repository này.

Checkpoint `checkpoints/pretrained/PushT_original_lowdim.ckpt` là trọng số pretrained của tác giả Diffusion Policy, được phân phối qua [artifact chính thức](https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt). Nó được lưu qua Git LFS để phục vụ tái lập và so sánh; không phải model do dự án này huấn luyện. Khi báo cáo kết quả, ghi rõ nguồn checkpoint và phân biệt với World Model/Diffusion Policy tự viết.

Bài báo tham khảo: Cheng Chi và cộng sự, [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137).

Bên dưới là nguyên văn LICENSE của repository gốc tại commit tham chiếu. Thông báo này giữ thông tin tác giả và giấy phép của thành phần gốc; không tự gán giấy phép đó cho toàn bộ phần code tự viết của dự án.

```text
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
