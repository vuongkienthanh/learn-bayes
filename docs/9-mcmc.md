---
title: "Chapter 9: Markov Chain Monte Carlo"
description: "Chương 9: Markov Chain Monte Carlo"
---

- [9.1 Quần đảo của Đức vua Markov](#a1)
- [9.2 Thuật toán Metropolis](#a2)
- [9.3 Hamiltonian Monte Carlo](#a3)
- [9.4 HMC đơn giản](#a4)
- [9.5 Chăm sóc và nuôi dưỡng chuỗi Markov](#a5)
- [9.6 Tổng kết](#a6)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import inspect
import math
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import lax, ops, random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation{% endhighlight %}</details>

Vào thế kỷ 20, các nhà khoa học và kỹ sư bắt đầu xuất bản sách về những con số ngẫu nhiên ([**HÌNH 9.1**](#f1)). Với những nhà khoa học từ thế kỷ trước, những cuốn sách này trông điên rồ. Với hầu hết lịch sử phương Tây, xác suất từng là kẻ thù. Điển hình như Rome, xác suất được hình tượng hoá bởi Fortuna, nữ thần của định mệnh, với vòng xoay của may rủi. Trái ngược với Fortuna là Minerva, nữ thần trí tuệ và tinh thông. Chỉ những người tuyệt vọng mới cầu nguyện Fortuna, trong khi mọi người cầu khẩn Minervà giúp đỡ. Chắc chắn khoa học là lĩnh vực của Minerva, một lãnh địa mà không có vai trò hữu ích của Fortuna.

Nhưng ở thế kỷ 20, Fortuna và Minerva đã thành cộng sự của nhau. Bây giờ một vài trong chúng ta chắc sẽ hoang mang bởi ý kiến rằng hiểu biết xác suất có thể giúp chúng ta đạt được tri trức. Mọi thứ từ dự báo thời tiết đến kinh tế đến sinh học tiến hoá là ưu thế bởi khoa học của tiến trình ngẫu nhiên. Nhà khoa học dựa vào những con số ngẫu nhiên để thiết kế thí nghiệm


## <center>9.1 Quần đảo của Đức vua Markov</center><a name="a1"></a>
## <center>9.2 Thuật toán Metropolis</center><a name="a2"></a>
## <center>9.3 Hamiltonian Monte Carlo</center><a name="a3"></a>
## <center>9.4 HMC đơn giản</center><a name="a4"></a>
## <center>9.5 Chăm sóc và nuôi dưỡng chuỗi Markov</center><a name="a5"></a>
## <center>9.6 Tổng kết</center><a name="a6"></a>


<a name="f1"></a>![](/assets/images/fig -.svg)
<details class="fig"><summary></summary>
{% highlight python %}{% endhighlight %}</details>

---

<details><summary>Endnotes</summary>
<ol class='endnotes'>
    <li></li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
</details>