---
title: "Chapter 6: The Haunted DAG & The Causal Terror"
description: "Chương 6: DAG bị ám và sự kinh hoàng của nhân quả"
---

- [6.1 Hiện tượng đa cộng tuyến](#1)
- [6.2 Sai lệch hậu điều trị](#2)
- [6.3 Sai lệch xung đột](#3)
- [6.4 Đối phó với hiện tượng nhiễu](#4)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import daft
import matplotlib.pyplot as plt
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel
import jax.numpy as jnp
from jax import lax, random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation{% endhighlight %}</details>

Có vẻ như những bài báo khoa học thời sự lại là những bài báo kém tin cậy nhất. Nó càng có khả năng giết bạn, nếu đúng, thì càng ít khả năng là nó đúng. Đề tài càng chán ngấy, thì kết quả của nó càng chính xác. Tại sao sự tương quan âm này được tin tưởng rộng rãi? Không có lý do gì để những nghiên cứu hay đề tài mà mọi người quan tâm lại cho kết quả kém tin cậy. Có thể nào những chủ đề phổ biến thu hút nhiều nhà nghiên cứu "dỏm", như mật ngọt hút ruồi?

Thật ra điều kiện cần cho tương quan âm này xuất hiện là những người đánh giá báo cáo khoa học (peer reviewer) quan tâm đến cả hai tính chất thời sự (newsworthiness) và tin cậy (trustworthiness). Nếu người đánh giá quan tâm đến hai tính chất trên, thì chính hành động chọn lọc đó làm cho bài báo thời sự nhất thành kém tin cậy nhất. Thực tế, rất khó để tưởng tượng rằng công việc đánh giá bài báo có cách nào tránh được hiện tượng này. Và, các bạn đọc thân mến, sự thật này sẽ giúp chúng ta hiểu những nguy hiểm đang rình rập của hồi quy đa biến.

Đây là một mô phỏng đơn giản để minh hoạ cho điểm này. Giả sử có một hội đồng duyệt bài nhận được 200 bài báo nộp lên. Thì giữa những bài đó, không có tương quan nào giữa tính tin cậy (độ chính xác, sự uyên thâm, khả năng thành công) và tính thời sự (giá trị phúc lợi xã hội, mối quan tâm công chúng). Hội đồng đo lường tính thời sự và tính tin cậy là như nhau. Sau đó họ xép hạng những bài nộp bởi tổng điểm của chúng và chọn ra 10% trên để cho quỹ.

<a name="f1"></a>![](/assets/images/fig 6-1.svg)
<details class="fig"><summary>Hình 6.1: Tại sao những bài cáo thời sự lại ít tin cậy nhất. 200 bài báo nộp lên được xếp hàng bằng tổng tính thời sự và tính tương cậy. 10% trên được chọn để cho quỹ. Trong khi không có tương quan trước khi chọn lọc, hai tiêu chuẩn này tương quan âm mạnh sau khi được chọn lọc. Tương quan ở đây là -0.65.</summary>
{% highlight python %}with numpyro.handlers.seed(rng_seed=1914):
    N, p = 200, 0.1
    nw = numpyro.sample("nw", dist.Normal().expand([N]))
    tw = numpyro.sample("tw", dist.Normal().expand([N]))
    s = nw + tw
    q = jnp.quantile(s, 1 - p)
    selected = jnp.where(s >= q, True, False)
plt.figure(figsize=(5,4))
ax=plt.gca(xlabel="tính thời sự", ylabel="tính tin cậy")
ax.scatter(nw,tw)
sns.regplot(x=nw[selected], y=tw[selected], ax=ax, color='C1', ci=None)
ax.annotate('chọn', (2,2))
ax.annotate('rớt', (0,-2)){% endhighlight %}</details>

Cuối phần này, tôi sẽ đưa code mô phỏng thí nghiệm này. [**HÌNH 6.1**](#f1) thể hiện toàn bộ mẫu các bài báo mô phỏng được nộp lên, và những bài báo được chọn là màu đỏ. Tôi vẽ thêm đường hồi quy tuyến tính đơn giản giữa các bài báo được chọn. Có một tương quan âm, -0.65 trong ví dụ này. Chọn lọc mạnh tạo ra tương quan âm giữa các tiêu chuẩn dùng để chọn lọc. Tại sao? Nếu cách duy nhất để vượt qua ngưỡng này để có nhiều điểm hơn, thì cách thông dụng là đạt điểm cao ở một tiêu chuẩn hơn là cả hai. Cho nên giữa các bài báo được chọn để cho quỹ, thì những bài báo thời sự có thể thực ra có tính tin cậy thấp hơn trung bình (nhỏ hơn 0 trong hình này). Tương tự những bài báo tin cậy cao thì có tính thời sự thấp hơn trung bình.

Hiện tượng này đã được nhận ra từ lâu. Nó đôi khi được gọi là **NGHỊCH LÝ BERKSON (BERKSON'S PARADOX)**. Nhưng sẽ dễ nhớ hơn nếu chúng ta gọi nó là *hiệu ứng chọn lọc-móp méo (selection-distortion effect)*. Một khi bạn quan tâm đến hiệu ứng này, bạn sẽ thấy nó tồn tại ở mọi nơi. Tại sao những nhà hàng ở vị trí tốt lại có thức ăn dở. Một cách tồn tại của nhà hàng với thức ăn không-ngon-lắm là phải ở vị trí tốt. Tương tự, nhà hàng với thức ăn ngon có thể sinh tồn được ở vị trí xấu. Chọn lọc-móp méo đã phá vỡ thành phố của chúng ta.

Vậy nó liên quan gì đến hồi quy đa biến (multiple regression)? Thật không may, mọi thứ. Chương trước giới thiệu hồi quy đa biến là một công cụ tuyệt vời để đánh tan mối tương quan giả tạo, cũng như làm rõ tương quan bị ẩn. Điều đó có lẽ củng cố rằng nên thêm tất cả mọi thứ vào mô hình và hãy để vị thánh hồi quy tự giải quyết.

Nhưng không, mô hình hồi quy đa biến không tự giải quyết được hết. Nó là một thiên thần, nhưng cũng là ác quỷ. Nó nói chuyện với giọng điệu đánh đố và sẽ trừng phạt chúng ta nếu cho nó một câu hỏi kém. Hiệu ứng chọn lọc-móp méo có thể xảy ra ngay trong hồi quy đa biến, bởi vì việc thêm biến dự đoán gây ra sự chọn lọc thống kê ngay trong mô hình, một hiện tượng với tên gọi không giúp ích được gì, **SAI LỆCH XUNG ĐỘT (COLLIDER BIAS)**. Nó làm cho chúng ta hiểu sai rằng, ví dụ, nhìn chung có một tương quan âm giữa tính thời sự và tính tin cậy, trong khi thực tế nó là hệ quả của việc đặt điều kiện trên các biến nào đó. Đây vừa là một sự thật gây bối rối vừa là một sự thật cực kỳ quan trọng để hiểu để dùng hồi quy một cách có trách nhiệm.

Chương này và tiếp theo đều về những thảm hoạ có thể xảy ra nếu chúng ta đơn thuần thêm biến vào hồi quy, mà không có ý tưởng rõ ràng về mô hình nhân quả. Trong chương này chúng ta sẽ khám phá ba hiểm hoạ khác nhau: hiện tượng đa cộng tuyến (multicollinearity), sai lệch hậu điều trị (post-treatment bias), và sai lệch xung đột (collider bias). Chúng ta sẽ kết thúc bằng kết nối tất cả những ví dụ này lại vào chung một khung quy trình có thể giúp chúng ta biến số nào phải và không được đưa vào mô hình để đạt được suy luận hợp lý. Nhưng khung quy trình này không làm giúp chúng ta bước quan trọng nhất: Nó không đưa ra mô hình hợp lý.

<div class="alert alert-dark">
<p><strong>Mô phỏng khoa học móp méo.</strong> Mô phỏng như này rất dễ thực hiện bằng code, một khi bạn đã thấy được vài ví dụ. Trong mô phỏng này, chúng ta sẽ rút mẫu ra từ vài tiêu chuẩn Gaussian ngẫu nhiên để có được một số lượng các bài báo nộp lên và sau đó chọn ra những bài có tổng điểm nằm trong 10% trên.</p>
<b>code 6.1</b>
{% highlight python %}with numpyro.handlers.seed(rng_seed=1914):
    N = 200  # num grant proposals
    p = 0.1  # proportion to select
    # uncorrelated newsworthiness and trustworthiness
    nw = numpyro.sample("nw", dist.Normal().expand([N]))
    tw = numpyro.sample("tw", dist.Normal().expand([N]))
    # select top 10% of combined scores
    s = nw + tw  # total score
    q = jnp.quantile(s, 1 - p)  # top 10% threshold
    selected = jnp.where(s >= q, True, False)
jnp.corrcoef(jnp.stack([tw[selected], nw[selected]], 0))[0, 1]{% endhighlight %}
<p>Tôi chọn ra seed cụ thể này để bạn có thể tái tạo kết quả trong <a href="#f1"><strong>HÌNH 6.1</strong></a>, nhưng nếu bạn chạy lại mô phỏng mà không cần đặt seed, bạn sẽ thấy không có gì đặc biệt trong seed mà tôi đã dùng.</p></div>

## <center>6.1 Hiện tượng đa cộng tuyến</center><a name="1"></a>

Ai cũng biết là có rất nhiều biến dự đoán tiềm năng để đưa vào một mô hình hồi quy. Trong trường hợp data sữa các loài khỉ, có đến 7 biến có sẵn để dự đoán bất kỳ cột nào được chọn là kết cục. Tại sao không xây dựng một mô hình chứa tất cả 7 biến vào? Có rất nhiều hiểm hoạ trong đó.

Hãy bắt đầu bằng nỗi lo lắng ít nhất của bạn, **HIỆN TƯỢNG ĐA CỘNG TUYẾN (MULTICOLLINEARITY)**. Đa cộng tuyến tức là có tồn tại một tương quan rất mạnh giữa hai hoặc nhiều biến. Giá trị tương quan thô không phải là cái đáng nói. Cái đáng nói là mối quan hệ, khi đặt điều kiện trên những biến khác trong mô hình. Hệ quả của đa cộng tuyến là phân phối posterior sẽ như đề nghị rằng không có biến nào liên quan đến kết quả đáng tin cậy cả, mặc dù tất cả các biến trong thực tế đều tương quan rất mạnh với kết cục.

Hiện tượng nhức đầu này xuất phát từ chi tiết cách mô hình hồi quy hoạt động. Thực tế, hiện tượng đa cộng tuyến không có gì sai. Mô hình vẫn cho dự đoán tốt. Bạn chỉ cảm thấy khốn khổ nếu muốn cố gắng hiểu nó. Hi vọng là sau khi bạn hiểu hiện tượng đa cộng tuyến, bạn sẽ nhìn chung hiểu mô hình hồi quy hơn.

Hãy bắt đầu bằng mô phỏng đơn giản. Sau đó chúng ta sẽ quay về data sữa các loài khỉ lần nữa và tìm ra đa cộng tuyến trong data thực.

### 6.1.1 Những cái chân đa cộng tuyến

Giả sử tưởng tượng muốn dự đoán chiều cao con người dựa vào biến dự đoán là chiều dài chân. Khẳng định rằng chiều dài chân tương quan dương với chiều cao cơ thể, hoặc ít nhất là trong mô phỏng sẽ như vậy. Dù thế nào, một khi bạn cho cả hai chân (trái và phải) vào mô hình, điều bất ngờ sẽ xảy ra.

Đoạn code sau sẽ mô phỏng chiều dài hai chân và chiều cao của 100 người. Với mỗi cá thể, đầu tiên thì có một chiều cao được mô phỏng từ phân phối Gaussian. Sau đó mỗi người có được một tỉ lệ chiều cao cho hai chân của họ, từ 0.4 đến 0.5. Sau cùng, mỗi chân được thêm gia vị với một ít sai số từ đo lường hoặc phát triển, để chân trái và chân phải không hoàn toàn giống nhau, giống như ngoài đời thực. Cuối cùng, đoạn code sẽ cho chiều cao và chiều dài hai chân trong chung một DataFrame.

<b>code 6.2</b>
```python
N = 100  # number of individuals
with numpyro.handlers.seed(rng_seed=909):
    # sim total height of each
    height = numpyro.sample("height", dist.Normal(10, 2).expand([N]))
    # leg as proportion of height
    leg_prop = numpyro.sample("prop", dist.Uniform(0.4, 0.5).expand([N]))
    # sim left leg as proportion + error
    leg_left = leg_prop * height + numpyro.sample(
        "left_error", dist.Normal(0, 0.02).expand([N])
    )
    # sim right leg as proportion + error
    leg_right = leg_prop * height + numpyro.sample(
        "right_error", dist.Normal(0, 0.02).expand([N])
    )
    # combine into data frame
    d = pd.DataFrame({"height": height, "leg_left": leg_left, "leg_right": leg_right})
```

Bây giờ hãy phân tích bộ data này, dự đoán kết cục `height` dựa trên cả hai biến dự đoán, `leg_left` và `leg_right`. Trước khi ước lượng posterior, tuy nhiên, hãy xem xét lại mong đợi của chúng ta. Trung bình, chiều dài hai chân của một cá nhân bằng 45% chiều cao của người đó (trong data mô phỏng). Cho nên chúng mong đợi hệ số beta mà đo lường quan hệ giữa một chân với chiều dài sẽ gần bằng chiều cao trung bình (10) chia cho 45% của chiều cao trung bình (4.5). Nó là $10/4.5 \approx 2.2$. Bây giờ hãy xem điều ngược lại sẽ xảy ra. Tôi sẽ dùng prior mơ hồ, nhưng prior kém, chỉ để chúng ta chắc chắn rằng prior không chịu trách nhiệm cho chuyện sắp xảy ra.

<b>code 6.3</b>
```python
def model(leg_left, leg_right, height):
    a = numpyro.sample("a", dist.Normal(10, 100))
    bl = numpyro.sample("bl", dist.Normal(2, 10))
    br = numpyro.sample("br", dist.Normal(2, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bl * leg_left + br * leg_right
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
m6_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    leg_left=d.leg_left.values,
    leg_right=d.leg_right.values,
    height=d.height.values,
)
p6_1, losses = svi.run(random.PRNGKey(0), 2000)
post = m6_1.sample_posterior(random.PRNGKey(1), p6_1, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a   0.81  0.34    0.80   0.27   1.31  1049.96   1.00
   bl   2.61  2.28    2.63  -1.06   6.26   813.11   1.00
   br  -0.59  2.28   -0.60  -4.41   2.96   805.68   1.00
sigma   0.67  0.05    0.67   0.60   0.74   968.52   1.00</samp>

Những con số trung bình và độ lệch chuẩn trong posterior trông thật điên rồ. Đây là một trường hợp trong đó dùng đồ thị biểu diễn posterior là hữu ích hơn, bởi vì nó thể hiện trung bình và khoảng 89% của posterior theo một cách cho phép chúng ta nhìn sơ là đã biết có gì đó sai ở đây:

<b>code 6.4</b>
```python
az.plot_forest(post, hdi_prob=0.89)
```

![](/assets/images/forest 6-1.svg)

Bạn hãy thử mô phỏng thêm vài lần nữa với seed khác. Nếu cả hai chân có chiều dài gần như bằng nhau, và chiều cao phải có tương quan mạnh với chiều dài chân, thì tại sao phân phối posterior kì lạ vậy? Kỹ thuật ước lượng posterior đúng chưa?

Kỹ thuật ước lượng đã hoạt động đúng, và nó cho kết quả posterior chính xác với câu hỏi của ta. Vấn đề đây là câu hỏi, nhớ lại rằng hồi quy đa biến trả lời câu hỏi: *Giá trị của việc biết mỗi biến dự đoán, khi chúng ta đã biết những biến dự đoán khác?* Trong trường hợp này, câu hỏi trở thành: *Giá trị của việc biết chiều dài mỗi chân, sau khi biết chiều dài chân còn lại?*

Đáp án cho câu hỏi lạ lùng này, cũng rất lạ lùng, nhưng hoàn toàn hợp logic. Trả lời đó là phân phối posterior ước lượng được, xem xét mọi trường hợp khả dĩ của các kết hợp của tham số và gán tính phù hợp tương đối cho mọi kết hợp đó, đặt điều kiện trên mô hình và data. Sẽ có ích hơn nếu nhìn vào phân phối posterior kết hợp của `bl` và `br`:

<b>code 6.5</b>
```python
post = m6_1.sample_posterior(random.PRNGKey(1), p6_1, (1000,))
az.plot_pair(post, var_names=["br", "bl"], scatter_kwargs={"alpha": 0.1})
```

<a name="f2"></a>![](/assets/images/fig 6-2.svg)
<details class="fig"><summary>Hình 6.2: Trái: Phân phối posterior cho quan hệ của mỗi chân với chiều cao, từ mô hình <code>m6_1</code>. Bởi vì cả hai biến đều chứa thông tin như nhau, posterior là một đường hẹp của các giá trị tương quan âm. Phải: phân phối posterior là tổng của hai tham số được tập trung vào quan hệ đúng giữa một trong hai chân với chiều cao.</summary>
{% highlight python %}fig, axs=plt.subplots(1,2,figsize=(8,4))
post = m6_1.sample_posterior(random.PRNGKey(1), p6_1, (1000,))
az.plot_pair(post, var_names=["br", "bl"], scatter_kwargs={"alpha": 0.1}, ax=axs[0])
sum_blbr = post["bl"] + post["br"]
az.plot_dist(sum_blbr, bw=0.01,ax=axs[1])
axs[1].set(xlabel="Tổng bl và br", ylabel="mật độ"){% endhighlight %}</details>

Kết quả ở bên trái [**HÌNH 6.2**](#f2). Phân phối posterior cho hai tham số này rất tương quan với nhau, và với mọi giá trị phù hợp của `bl` và `br` nằm trên một khe hẹp. Khi `bl` lớn, thì `br` phải nhỏ. Chuyện xảy ra ở đây là bởi vì cả hai biến số chân đều chứa đựng cùng một thông tin, nếu bạn ép buộc cho hai biến vào chung mô hình, thì sẽ có vô số cặp `bl` và `br` để tạo ra cùng một kết quả.

Một cách khác về nhìn hiện tương này là bạn đã dựng mô hình như sau:

$$\begin{aligned}
y_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta_1 x_i + \beta_2 x_i\\
\end{aligned}$$

Biến $y$ là kết cục, như chiều cao trong ví dụ, và $x$ là một biến dự đoán, giống như chiều dài chân trong ví dụ. Ở đây, biến $x$ được dùng 2 lần, là một ví dụ hoàn hảo về rắc rối khi sử dụng chiều dài cả hai chân. Dưới góc nhìn của golem, mô hình của $\mu_i$ sẽ là:

$$ \mu_i = \alpha + (\beta_1 + \beta_2)x_i $$

Tôi đã cho biến $x$ là nhân tử chung của biểu thức, tham số $\beta_1$ và $\beta_2$ không thể bị tách rời, bởi vì chúng không bao giờ tách rời và ảnh hưởng cùng lúc vào trung bình $\mu$. Thứ ảnh hưởng nó là tổng của tham số, $\beta_1 + \beta_2$. Cho nên phân phối posterior cho kết quả một khoảng rất rộng của các kết hợp $\beta_1$ và $\beta_2$ mà tổng của chúng thể hiện tương quan thực sự giữa $x$ và $y$.

Và phân phối posterior của ví dụ mô phỏng này là thực hiện điều đó chính xác: Nó đã tạo ta một ước lượng tốt cho tổng của `bl` và `br`. Bây giờ đây là cách để bạn tính được phân phối posterior của tổng đó, Nó được thể hiện hình trên bên phải, và vẽ nó lên:

<b>code 6.6</b>
```python
sum_blbr = post["bl"] + post["br"]
az.plot_kde(sum_blbr, label="Tổng bl và br")
```

Và kết quả là biểu đồ mật độ được thể hiện ở bên phải của [**HINH 6.2**](#f3). Trung bình của tổng ấy nằm ở khu hàng xóm bên phải, giá trị lớn hơn 2 một chút, và độ lệch chuẩn cũng nhỏ hơn nhiều so với từng thành phần riêng biệt của tổng, `bl` và `br`. Nếu bạn hồi quy chỉ với một biến chiều dài chân, bạn cũng sẽ có posterior tương tự.

<b>code 6.7</b>
```python
def model(leg_left, height):
    a = numpyro.sample("a", dist.Normal(10, 100))
    bl = numpyro.sample("bl", dist.Normal(2, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bl * leg_left
    numpyro.sample("height", dist.Normal(mu, sigma), obs=height)
m6_2 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_2,
    optim.Adam(1),
    Trace_ELBO(),
    leg_left=d.leg_left.values,
    height=d.height.values,
)
p6_2, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_2.sample_posterior(random.PRNGKey(1), p6_2, (1000,))
print_summary(post, 0.89, False)
```
<samp>
       mean   std  median  5.5%  94.5%   n_eff  r_hat
    a  0.83  0.35    0.84  0.25   1.35  931.50   1.00
   bl  2.02  0.08    2.02  1.91   2.15  940.42   1.00
sigma  0.67  0.05    0.67  0.60   0.75  949.09   1.00</samp>

Con số 2.02 đó là gần giống như giá trị trung bình của `sum_blbr`.

Bài học ở đây là: khi hai biến dự đoán tương quan rất mạnh với nhau (khi đặt điều kiện trên những biến khác trong mô hình), việc thêm cả hai vào mô hình sẽ dẫn đến sự bối rối. Phân phối posterior không sai, trong những trường hợp này. Nó nói cho bạn biết rằng câu hỏi cũng bạn không thể được trả lời bằng những data này. Và nó là một điều tốt cho mô hình để nói rằng, nó không thể trả lời câu hỏi của bạn. Và nếu bạn quan tâm đến dự đoán, bạn sẽ thấy rằng mô hình chân vẫn cho dự đoán chính xác. Nó chỉ không trả lời được là chân nào quan trọng hơn.

Ví dụ chân này đơn giản và dễ hiểu. Nhưng nó đơn thuần là thống kê. Chúng ta vẫn chưa đặt câu hỏi nhân quả nghiêm túc nào ở đây. Hãy thử một ví dụ nhân quả thú vị hơn sau đây.

### 6.1.2 Sữa đa cộng tuyến

Trong ví dụ chân, nó rất dễ để thấy rằng đưa cả hai chân vào mô hình là ngu ngốc. Nhưng vấn đề xuất hiện trong data thực là bạn không thể lường trước được sự xung đột giữa các biến dự đoán có tương quan cao. Và do đó chúng ta có thể hiểu sai phân phối posterior để nói rằng không biến nào là quan trọng. Trong phần này, hãy nhìn một ví dụ về vấn đề này với data thực.

Hãy quay lại data sữa các loài khỉ ở chương trước:

<b>code 6.8</b>
```python
d = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv?raw=true", sep=";")
d["K"] = d["kcal.per.g"].pipe(lambda x: (x - x.mean()) / x.std())
d["F"] = d["perc.fat"].pipe(lambda x: (x - x.mean()) / x.std())
d["L"] = d["perc.lactose"].pipe(lambda x: (x - x.mean()) / x.std())
```

Trong ví dụ này, chúng ta quan tâm đến hai biến `perc.fat` (phần trăm chất béo) và `perc.lactose` (phần trăm lactose). Chúng ta sẽ dùng chúng để dựng mô hình cho `kcal.per.g` (tổng năng lượng). Đoạn code trên đã chuẩn hoá ba biến này. Bạn sẽ dùng ba biến này để khám phá một trường hợp tự nhiên về đa cộng tuyến. Chú ý rằng không có giá trị mất `NaN` trong data, cho nên không cần phải trích xuất những trường hợp đầy đủ. Nhưng bạn có thể an tâm với `SVI`, bởi vì không giống như những phần mềm tự động, nó sẽ không im lặng với những trường hợp bị mất dữ liệu.

Bắt đầu bằng mô hình `kcal.per.g` như là một hàm số của `perc.fat` và `perc.lactose`, nhưng ở hồi quy hai biến. Nhìn lại Chương 5, để thảo luận về những prior này.

<b>code 6.9</b>
```python
# kcal.per.g regressed on perc.fat
def model(F, K):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bF = numpyro.sample("bF", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bF * F
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m6_3 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_3, optim.Adam(1), Trace_ELBO(), F=d.F.values, K=d.K.values)
p6_3, losses = svi.run(random.PRNGKey(0), 1000)
# kcal.per.g regressed on perc.lactose
def model(L, K):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bL = numpyro.sample("bL", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bL * L
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m6_4 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_4, optim.Adam(1), Trace_ELBO(), L=d.L.values, K=d.K.values)
p6_4, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_3.sample_posterior(random.PRNGKey(1), p6_3, (1000,))
print_summary(post, 0.89, False)
post = m6_4.sample_posterior(random.PRNGKey(1), p6_4, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a   0.01  0.08    0.01  -0.13   0.12   931.50   1.00
   bF   0.86  0.09    0.86   0.73   1.01  1111.41   1.00
sigma   0.46  0.06    0.46   0.37   0.57   940.36   1.00
        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a   0.01  0.07    0.01  -0.10   0.11   931.50   1.00
   bL  -0.90  0.07   -0.90  -1.01  -0.78  1111.89   1.00
sigma   0.39  0.05    0.39   0.31   0.48   957.39   1.00</samp>

Phân phối posterior của `bF` và `bL` chính là hình ảnh gương soi của nhau. Posterior trung bình của `bF` là dương trong khi trung bình của `bL` là âm. Cả hai đều có phân phối hẹp và nằm hoàn toàn ở hai bên của zero. Giả sử cả hai biến dự đoán đều có tương quan mạnh với kết cục, chúng ta đã có thể kết luận rằng cả hai biến đều là biến dự đoán đáng tin cậy cho tổng năng lượng sữa, ở các loài. Nhiều chất béo hơn, thì nhiều kilocalory hơn trong sữa. Lactose nhiều hơn, thì ít kilocalory trong sữa. Nhưng hãy xem chuyện gì sẽ xảy ra khi cho cả hai biến dự đoán vào chung mô hình hồi quy:

<b>code 6.10</b>
```python
def model(F, L, K):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bF = numpyro.sample("bF", dist.Normal(0, 0.5))
    bL = numpyro.sample("bL", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bF * F + bL * L
    numpyro.sample("K", dist.Normal(mu, sigma), obs=K)
m6_5 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_5, optim.Adam(1), Trace_ELBO(), F=d.F.values, L=d.L.values, K=d.K.values)
p6_5, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_5.sample_posterior(random.PRNGKey(1), p6_5, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a  -0.02  0.07   -0.03  -0.13   0.07  1049.96   1.00
   bF   0.25  0.19    0.25  -0.05   0.56   823.80   1.00
   bL  -0.67  0.19   -0.67  -0.99  -0.37   875.48   1.00
sigma   0.38  0.05    0.38   0.30   0.46   982.83   1.00</samp>

Bây giờ trung bình posterior của cả hai `bF` và `bL` trở nên gần zero hơn. Và độ lệch chuẩn của hai tham số thì rộng gấp hai lần so với mô hình hai biến (`m6_3` và `m6_4`).

Đây là một hiện tượng thống kê giống như ví dụ chiều dài chân. Những gì xảy ra ở đây là biến `perc.fat` và `perc.lactose` đều chứa cùng một thông tin. Chúng hầu là thay thế của nhau. Kết quả là, khi bạn cho cả hai biến vào mô hình, phân phối posterior sẽ mô tả một khe rất dài các kết hợp `bF` và `bL` mà có tính phù hợp như nhau. Trong trường hợp chất béo và lactose, hai biến này tạo ra cùng một trục biến thiên. Các đơn giản nhất để nhận ra là dùng biểu đồ bắt cặp.

<b>code 6.11</b>
```python
az.plot_pair(d[["kcal.per.g", "perc.fat", "perc.lactose"]].to_dict("list"))
```

<a name="f3"></a>![](/assets/images/fig 6-3.svg)
<details class="fig"><summary>Hình 6.3: Biểu đồ bắt cặp của các biến số tổng năng lượng, phần trăm chất béo, và phần trăm lactose từ data sữa các loài khỉ. Phần trăm chất béo và phần trăm lactose có tương quan âm mạnh với nhau, cung cấp hầu như cùng một thông tin.</summary>
{% highlight python %}az.plot_pair(d[["kcal.per.g", "perc.fat", "perc.lactose"]].to_dict("list"), figsize=(5,5)){% endhighlight %}</details>

Biểu đồ này được thể hiện trong [**HÌNH 6.3**](#f3). Tên các biến được dán nhãn ở các trục, tạo thành các nhãn của biểu đồ phân tán. Chú ý rằng phần trăm chất béo tương quan dương với kết cục, trong khi phần trăm lactose tương quan âm với nó. Bây giờ nhìn vào biểu đồ phân tán bên phải. Biểu đồ là phân tán giữa thành phần chất béo (trục hoành) và thành phần lactose (trục tung). Chú ý rằng các điểm được xếp hàng hầu như trên một đường thẳng. Hai biến này là tương quan âm với nhau, và mạnh đến mức một trong chúng là dư thừa. Một trong hai là hữu ích khi dự đoán `kcal.per.g`, nhưng không biến nào là có ích *một khi bạn đã biết biến còn lại*.

Trong giới khoa học, bạn có thể đã từng gặp nhiều phương pháp khác nhau để tránh hiện tượng đa cộng tuyến. Một ít trong số đó là dùng suy luận nhân quả. Vài lĩnh vực thì dạy học sinh khảo sát tương quan từng cặp trước khi fit mô hình, để xác định và loại bỏ những biến dự đoán có tương quan mạnh. Phương pháp này là sai. Bản thân tương quan từng cặp không phải là vấn đề. Vấn đề là mối liên quan có điều kiện - không phải tương quan. Và mặc dù thế, cách đúng nhất là dựa vào cái gì gây ra hiện tượng đa cộng tuyến. Mối quan hệ sẵn có trong data là không đủ để quyết định bước tiếp theo.

Những gì xảy ra trong ví dụ sữa có thể là có sự đánh đổi chính yếu giữa các thành phần trong sữa mà cơ thể giống cái phải tuân theo. Nếu người mẹ cho bú thường xuyên, sữa có thể chứa nhiều nước và ít năng lượng. Sữa như vậy có nồng độ đường (lactose) cao. Nếu thay vào đó người mẹ cho bú ít hơn, bằng những cữ nhỏ, thì sữa phải giàu năng lượng. Sữa như vậy có nhiều chất béo. Điều đó suy ra mô hình nhân quả như sau:

![](/assets/images/dag 6-1.svg)

Trung tâm của sự đánh đổi này quyết định độ đặc cần thiết của sữa, $D$. Chúng ta chưa quan sát được biến này, nên được đánh dấu thành một điểm. Sau đó chất béo $F$ và lactose $L$ được suy ra. Cuối cùng, tỉ lệ của $F$ và $L$ quyết định kilocalory, $K$. Nếu chúng ta có thể đo lường được $D$, hoặc có một mô hình tiến hoá hay kinh tế nào đó dự đoán được giá trị đó dựa trên những khía cạnh khác của loài, thì điều đó tốt hơn là nghịch các mô hình hồi quy.

Hiện tượng đa cộng tuyến là một thành viên của những vấn đề liên quan với việc fit mô hình, những vấn đề đó còn gọi là **KHÔNG CÓ KHẢ NĂNG NHẬN DIỆN (NON-IDENTIFIABILITY)**. Khi một tham số không có khả năng nhận diện được, có nghĩa là cấu trúc của data và mô hình làm cho không có khả năng ước lượng được tham số đó. Thông thường, vấn đề này xảy ra do lỗi thiết kế mô hình sai, nhưng rất nhiều mô hình quan trọng vẫn chứa tham số không hoặc khó nhận diện được, cho dù thiết kế đúng hoàn toàn. Tự nhiên không nợ chúng ta những suy luận đơn giản, ngay khi mô hình là đúng.

Nói chung, không đảm bảo lúc nào data cũng có đầy đủ thông tin về tham số quan tâm. Nếu nó thực sự đúng, suy luận bayes sẽ cho phân phối posterior gần giống với prior. So sánh prior với posterior có thể là một bước đi hợp logic để nhìn thấy mô hình đã trích xuất được bao nhiêu thông tin từ data về tham số. Khi posterior và prior giống nhau, không có nghĩa là phép tính đã sai - bạn vẫn có được đáp án đúng cho câu hỏi của bạn. Nhưng nó có thể hướng dẫn bạn đặt câu hỏi tốt hơn.

<div class="alert alert-info">
<p><strong>Đảm bảo sự nhận diện; hiểu là tuỳ bạn.</strong> Về mặt kỹ thuật, <i>sự nhận diện</i> không phải là vấn đề của mô hình Bayes. Lý do là chỉ cần phân phối posterior là đúng - có tích phân là 1 - thì tất cả những tham số là nhận diện được. Nhưng điều này không có nghĩa là bạn sẽ hiểu được phân phối posterior. Cho nên tốt hơn nên nói là tham số <i>nhận diện yếu</i> trong bối cảnh Bayes. Nhưng sự khác nhau này chỉ mang tính kỹ thuật. Sự thật là ngay cả với DAG nói một hiệu ứng nhân quả là nhận diện được, nó có thể không nhận diện bằng thống kê. Chúng ta phải cố gắng hết sức trong thống kê, cũng như khi chúng ta thiết kê mô hình.</p></div>

<div class="alert alert-dark">
<p><strong>Mô phỏng dữ liệu đa cộng tuyến.</strong> Để xem posterior sai lệch như thế nào khi tương quan giữa hai biến dự đoán tăng lên, hãy dùng mô phỏng. Đoạn code này cho một hàm số mà tạo ra các biến dự đoán tương quan với nhau, fit mô hình, và trả về độ lệch chuẩn của phân phối posterior của slope liên quan <code>perc.fat</code> với <code>kcal.per.g</code>. Sau đó đoạn code này gọi hàm này và lặp lại nhiều lần, với mức độ tương quan khác nhau trong đầu vào, và thu thập kết quả.</p>
<b>code 6.12</b>
{% highlight python %}def sim_coll(i, r=0.9):
    sd = jnp.sqrt((1 - r ** 2) * jnp.var(d["perc.fat"].values))
    x = dist.Normal(r * d["perc.fat"].values, sd).sample(random.PRNGKey(3 * i))
    def model(perc_fat, kcal_per_g):
        intercept = numpyro.sample("intercept", dist.Normal(0, 10))
        b_perc_flat = numpyro.sample("b_perc.fat", dist.Normal(0, 10))
        b_x = numpyro.sample("b_x", dist.Normal(0, 10))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(2))
        mu = intercept + b_perc_flat * perc_fat + b_x * x
        numpyro.sample("kcal.per.g", dist.Normal(mu, sigma), obs=kcal_per_g)
    m = AutoLaplaceApproximation(model)
    svi = SVI(
        model,
        m,
        optim.Adam(0.01),
        Trace_ELBO(),
        perc_fat=d["perc.fat"].values,
        kcal_per_g=d["kcal.per.g"].values,
    )
    params, losses = svi.run(random.PRNGKey(3 * i + 1), 20000, progress_bar=False)
    samples = m.sample_posterior(random.PRNGKey(3 * i + 2), params, (1000,))
    vcov = jnp.cov(jnp.stack(list(samples.values()), axis=0))
    stddev = jnp.sqrt(jnp.diag(vcov))  # stddev of parameter
    return dict(zip(samples.keys(), stddev))["b_perc.fat"]
def rep_sim_coll(r=0.9, n=100):
    stddev = lax.map(lambda i: sim_coll(i, r=r), jnp.arange(n))
    return jnp.nanmean(stddev)
r_seq = jnp.arange(start=0, stop=1, step=0.01)
stddev = lax.map(lambda z: rep_sim_coll(r=z, n=100), r_seq)
plt.plot(r_seq, stddev)
plt.xlabel("correlation"){% endhighlight %}
<p>Nên với mỗi hệ số tương quan trong <code>r_seq</code>, code này tạo ra 100 hồi quy và trả về độ lệch chuẩn trung bình từ nó. Code này dùng prior phẳng, tức là prior kém. Cho nên nó phóng đại hiệu ứng của biến cộng tuyến. Khi bạn sử dụng prior chứa thông tin, thì sự khuếch đại của độ lệch chuẩn có thể chậm lại.</p>
</div>

## <center>6.2 Sai lệch do biến hậu điều trị (Post-treatment bias)</center><a name="2"></a>

Trong suy luận, ta thường gặp những *sai lệch do thiếu biến*, đã được mô tả ở chương trước. Còn sai lệch do dư biến thì ít được quan tâm hơn. Nhưng sai lệch do dư biến là có tồn tại, ngay cả thí nghiệm ngẫu nhiên cẩn thận có thể bị phá hoại cũng như nghiên cứu quan sát không kiểm soát. Thêm biến một cách mù quáng vào *causal salad* không bao giờ là ý tưởng tốt.

*Sai lệch do dư biến* cũng có rất nhiều loại. Đầu tiên là *sai lệch do biến hậu điều trị*. Nó có trong mọi loại nghiên cứu. Ví dụ bạn đang trồng cây tại một nhà kính. Bạn muốn biết sự khác nhau về phát triển giữa nhiều loại "điều trị" phân bón kháng nấm khác nhau trên cây, bơi vì nấm thường cản trở sự phát triển của cây. Lúc bắt đầu, cây được giao hạt và nảy mầm, thu thập dữ liệu chiều cao ban đầu của cây. Các "điều trị" phân bón kháng nấm được ghi nhận. Kết quả là chiều cao cuối cùng và sự hiện diện của nấm. Có 4 biến: chiều cao ban đầu, chiều cao cuối cùng, điều trị, hiện diện của nấm.

Chiều cao cuối cùng là outcome quan tâm. Nhưng ta cần thêm biến nào khác trong mô hình? Nếu mục tiêu là suy luận nhân quả về hiệu quả điều trị, bạn không nên bao gồm biến nấm. bởi vì nó là hiệu ứng hậu điều trị. Ta có thể mô phỏng data như sau.

```python
with numpyro.handlers.seed(rng_seed=71):
    # number of plants
    N = 100

    # simulate initial heights
    h0 = numpyro.sample("h0", dist.Normal(10, 2).expand([N]))

    # assign treatments and simulate fungus and growth
    treatment = jnp.repeat(jnp.arange(2), repeats=N // 2)
    fungus = numpyro.sample(
        "fungus", dist.Binomial(total_count=1, probs=(0.5 - treatment * 0.4))
    )
    h1 = h0 + numpyro.sample("diff", dist.Normal(5 - 3 * fungus))

    # compose a clean data frame
    d = pd.DataFrame({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})
print_summary(dict(zip(d.columns, d.T.values)), 0.89, False)
```

|           |  mean |  std | median |  5.5% | 94.5% | n_eff | r_hat |
|    fungus |  0.31 | 0.46 |   0.00 |  0.00 |  1.00 | 18.52 |  1.17 |
|        h0 |  9.73 | 1.95 |   9.63 |  7.05 | 13.33 | 80.22 |  0.99 |
|        h1 | 13.72 | 2.47 |  13.60 | 10.73 | 18.38 | 43.44 |  1.08 |
| treatment |  0.50 | 0.50 |   0.50 |  0.00 |  1.00 |  2.64 |   inf |

>**Sai lệch do biến hậu điều trị** và sự nguy hiểm của nó được khám phá từ lâu. Người ta được dạy rằng khi đặt điều kiện lên biến hậu điều trị có nhiều nguy cơ, còn biến tiền điều trị thì an toàn. Nhưng nó không phải nguyên tắc chung. Biến tiền điều trị cũng có thể gây sai lệch. Ta sẽ thấy điều đó sau chương này.

### 6.2.1 Thiết kế prior

Khi thiết kế mô hình, bạn không có và không biết bộ máy tạo data như trên. Nhưng bạn có kiến thức nền tảng để xây dựng mô hình.

Chúng ta biết rằng chiều cao tại thời điểm $t=1$ sẽ cao hơn so với thời điểm $t=0$. Cho nên ta đặt parameter sẽ là tỉ lệ giữa 2 chiều cao ở 2 thời điểm đó. Trong trường hợp đơn giản, ta chỉ quan tâm đến biến chiều cao.

$$\begin{matrix}
h_{1,i} & \propto & \text{Normal}(\mu_i, \sigma) \\
\mu_i  &=& h_{0,i} \times p \\
p & \propto & \text{Log-Normal} (0 ,0.25) \\
\end{matrix}$$

Trong đó, tỉ lệ $p$ là parameter cần quan tâm. Nếu $p=2$, tức là chiều cao tăng gấp đôi. Nếu đặt prior của $p$ xung quanh $p=1$, nghĩa là ta mong đợi không có sự thay đổi chiều cao. Nhưng ta nên cho phép $p<1$, trong trường hợp thí nghiệm gặp trục trặc và diệt hết các cây. Ta cũng phải đảm bảo khả năng $p>0$, bởi vì nó là tỉ lệ. Trong các chương trước, ta dùng phân phối Log-Normal, bởi vì nó luôn dương. Lần này ta cũng vậy, prior sẽ trông giống như sau:

```python
sim_p = dist.LogNormal(0, 0.25).sample(random.PRNGKey(0), (int(1e4),))
print_summary({"sim_p": sim_p}, 0.89, False)
```

|       | mean |  std | median | 5.5% | 94.5% |   n_eff | r_hat |
| sim_p | 1.04 | 0.27 |   1.00 | 0.63 |  1.44 | 9936.32 |  1.00 |

Với prior này, ta mong đợi chiều cao bị nhỏ lại 40% hoặc cao hơn 50%. Ta fit mô hình để xem hiệu ứng phát triển trung bình trong thí nghiệm này.

```python
def model(h0, h1):
    p = numpyro.sample("p", dist.LogNormal(0, 0.25))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = h0 * p
    numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)


m6_6 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_6, optim.Adam(0.5), Trace_ELBO(), h0=d.h0.values, h1=d.h1.values)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_6 = svi.get_params(state)
post = m6_6.sample_posterior(random.PRNGKey(1), p6_6, (1000,))
print_summary(post, 0.89, False)
```

|       | mean |  std | median | 5.5% | 94.5% |   n_eff | r_hat |
|     p | 1.40 | 0.02 |   1.40 | 1.37 |  1.43 |  994.31 |  1.00 |
| sigma | 1.85 | 0.13 |   1.85 | 1.64 |  2.05 | 1020.42 |  1.00 |

Chiều cao tăng trung bình khoảng 40%. Bây giờ ta thêm biến điều trị và biến hiện diện nấm vào mô hình, dựa trên mục đích đo lường hiệu ứng của điều trị và hiện diện nấm. Parameter của những biến này vẫn sẽ ở dạng tỉ lệ. Ta dựng mô hình tuyến tính theo $p$.

$$\begin{matrix}
h_{1,i} & \propto & \text{Normal}(\mu_i, \sigma) \\
\mu_i  &=& h_{0,i} \times p \\
p &=& \alpha +\beta_T T_i +\beta_F F_i \\
\alpha  & \propto & \text{Log-Normal}(0, 0.25) \\
\beta_T & \propto & \text{Normal} (0, 0.5)\\
\beta_F & \propto & \text{Normal} (0, 0.5)\\
\sigma  & \propto & \text{Exponential} (1) \\
\end{matrix}$$

Tỉ lệ tăng trưởng $p$ bây giờ là hàm số của các biến dự đoán. Prior của các slope khá là phẳng. Khoảng 95% của các prior nằm ở giữa -1 (giảm 100%) và +1 (tăng 100%) và 2/3 mật độ xác suất nằm giữa -0.5 và +0.5. 

```python
def model(treatment, fungus, h0, h1):
    a = numpyro.sample("a", dist.LogNormal(0, 0.2))
    bt = numpyro.sample("bt", dist.Normal(0, 0.5))
    bf = numpyro.sample("bf", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    p = a + bt * treatment + bf * fungus
    mu = h0 * p
    numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)


m6_7 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_7,
    optim.Adam(0.3),
    Trace_ELBO(),
    treatment=d.treatment.values,
    fungus=d.fungus.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_7 = svi.get_params(state)
post = m6_7.sample_posterior(random.PRNGKey(1), p6_7, (1000,))
print_summary(post, 0.89, False)
```

       |  mean |  std | median |  5.5% | 94.5% |   n_eff | r_hat |
     a |  1.47 | 0.02 |   1.47 |  1.44 |  1.51 | 1049.13 |  1.00 |
    bf | -0.28 | 0.03 |  -0.28 | -0.33 | -0.24 |  911.14 |  1.00 |
    bt |  0.01 | 0.03 |   0.01 | -0.03 |  0.06 | 1123.14 |  1.00 |
 sigma |  1.25 | 0.08 |   1.25 |  1.11 |  1.37 |  982.60 |  1.00 |

Parameter $a$ thì giống như $p$ phần trước. Nó có cùng posterior. Posterior của hiệu ứng điều trị $\beta_T$ là zero và khoảng tin cậy hẹp. Việc điều trị không liên quan với sự phát triển. Nấm thì có vẻ gây giảm sự phát triển. Cho rằng ta biết việc điều trị giúp cho sự phát triển, vì vốn dĩ ta đã tự tạo ra data mô phỏng, thế chuyện gì đã xảy ra?

### 6.2.2 Bị chặn bởi kết quả điều trị

Vấn đề là sự hiện diện của nấm là kết quả của việc điều trị, hay nấm là biến hậu điều trị. Khi chúng ta kiểm soát biến nấm, câu hỏi dành cho mô hình là: *khi chúng ta biết có hay không có sự hiện diện của nấm, việc điều trị có ảnh hưởng gì không?* Câu trả lời là "không", bởi vì việc điều trị ảnh hưởng sự phát triển thông qua việc giảm nhiễm nấm. Nhưng cái ta quan tâm ở đây là hiệu ứng của việc điều trị lên sự phát triển chiều cao. Đúng hơn, mô hình phải loại bỏ biến hiện diện nấm.

```python
def model(treatment, h0, h1):
    a = numpyro.sample("a", dist.LogNormal(0, 0.2))
    bt = numpyro.sample("bt", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    p = a + bt * treatment
    mu = h0 * p
    numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)


m6_8 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_8,
    optim.Adam(1),
    Trace_ELBO(),
    treatment=d.treatment.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_8 = svi.get_params(state)
post = m6_8.sample_posterior(random.PRNGKey(1), p6_8, (1000,))
print_summary(post, 0.89, False)
```

|       | mean |  std | median | 5.5% | 94.5% |  n_eff | r_hat |
|     a | 1.33 | 0.02 |   1.33 | 1.29 |  1.37 | 930.82 |  1.00 |
|    bt | 0.13 | 0.04 |   0.12 | 0.08 |  0.19 | 880.02 |  1.00 |
| sigma | 1.73 | 0.12 |   1.73 | 1.55 |  1.94 | 948.82 |  1.00 |

Giờ thì hiệu ứng của điều trị rõ ràng hơn, và nó đáng lẽ như vậy. Ta nên đặt biến $h_0$ vào mô hình để thấy sự khác biệt trước và sau điều trị, nó có thể che đậy hiệu ứng nhân quả của điều trị. Nhưng biến hậu điều trị bản chất là che đậy luôn việc điều trị. Điều đó không có nghĩa là bạn không muốn cả hai biến trong cùng mô hình. Việc hệ số điều trị bị zero do biến hậu điều trị, giải thích rằng điều trị hoạt động đúng với mong đợi. Nó nói cho chúng ta biết về cơ chế. Nhưng nếu ta muốn suy luận đúng về điều trị thì phải loại bỏ biến hậu điều trị.

### 6.2.3 Nấm và tách biệt chiều hướng (d-separation)

Để đi sâu hơn, ta sẽ dùng tới DAG.

![](/assets/images/fig 6-6.png)

Điều trị $T$ ảnh hưởng đến sự hiện diện của nấm $F$, sau đó ảnh hưởng đến chiều cao của cây ở thời điểm 1 là $H_1$. Và $H_1$ cũng bị ảnh hưởng bởi chiều cao cây ở thời điểm 0, $H_0$. Nó là DAG của chúng ta. Nếu chúng ta thêm biến $F$, biến hậu điều trị, vào mô hình, thì chúng ta đã chặn con đường từ $T$ đến biến outcome.

Một cách nói khác, là việc kiểm soát biến $F$ đã tạo **D-SEPARATION**. "d" là *directional*, chiều hướng. Nó có nghĩa là 2 biến độc lập với nhau, tức là không có con đường nào nối giữa chúng. Trong trường hợp này, $H_1$ với $T$ là d-separation, nếu đặt điều kiện lên $F$. Có thể liệt kê tất cả các quan hệ độc lập thông qua package `causalgraphicalmodel`.

```python
plant_dag = CausalGraphicalModel(
    nodes=["H0", "H1", "F", "T"], edges=[("H0", "H1"), ("F", "H1"), ("T", "F")]
)
plant_dag.get_all_independence_relationships()
```

$$ F \perp H_0  \\
H_0 \perp T  \\
H_1 \perp T \| F $$

Có 3 quan hệ độc lập ở đây. Ta tập trung vào quan hệ thứ 3. Nhưng quan hệ 1 và 2 cũng cho ta hiểu hơn về mô hình nhân quả, như ở đây là chiều cao ban đầu $H_0$ không phụ thuộc vào điều trị và hiện diện của nấm, mà không cần đặt điều kiện nào cả.

Điều hiển nhiên là biến hậu điều trị cũng là một vấn đề trong thiết kế nghiên cứu quan sát cũng như nghiên cứu thực nghiệm. Trong thực nghiệm, ta dễ dàng nhận biết được biến tiền điều trị và biến hậu điều trị, trong nghiên cứu quan sát thì khó phân biệt chúng hơn. Nhưng trong thực nghiệm cũng phải cẩn thận cạm bẫy, như trong DAG sau đây:

![](/assets/images/fig 6-7.png)

Trong DAG này, biến $T$ ảnh hưởng đến $F$, nhưng $F$ không ảnh hưởng chiều cao cây, có lẽ cây này không hề quan tâm có loại nấm này luôn. Biến mới là $M$, độ ẩm - moisture. Nó ảnh hưởng $H_1$ và $F$, nhưng được khoanh tròn, tức là không được quan sát. Hồi quy $H_1$ trên $T$ sẽ cho thấy không có quan hệ giữa $T$ và $H_1$. Nhưng khi cho $F$ vào, lập tức mối quan hệ này xuất hiện.

```python
with numpyro.handlers.seed(rng_seed=71):
    N = 1000
    h0 = numpyro.sample("h0", dist.Normal(10, 2).expand([N]))
    treatment = jnp.repeat(jnp.arange(2), repeats=N // 2)
    M = numpyro.sample("M", dist.Bernoulli(probs=0.5).expand([N]))
    fungus = numpyro.sample(
        "fungus", dist.Binomial(total_count=1, probs=(0.5 - treatment * 0.4))
    )
    h1 = h0 + numpyro.sample("diff", dist.Normal(5 + 3 * M))
    d2 = pd.DataFrame({"h0": h0, "h1": h1, "treatment": treatment, "fungus": fungus})
```

Tại sao lại xảy ra điều này? Đó là do hiệu ứng collider. 

## <center>6.3 Sai lệch do biến xung dột (Collider bias)</center><a name="3"></a>

Quay lại ví dụ đầu tiên về tương quan âm giữa tính *thời sự* và tính *tin cậy* của các bài báo khoa học trong quá trình chọn lựa đăng tạp chí. Hiện tượng trên còn được gọi là *collider bias*.

Xem DAG dưới đây, $S$ là chọn lựa - selection, $T$ là tính tin cậy - trustworthy, $N$ là tính thời sự - newsworthy.

![](/assets/images/fig 6-8.png)

Có 2 mũi tên cho vào $S$ nên nó là một collider. Nguyên lý trong collider rất dễ hiểu, khi bạn đặt điều kiện vào collider, nó tạo ra tương quan thống kê giữa các nguyên nhân. Trong trường hợp này, khi bạn biết được $S$, khi biết thêm $T$ sẽ biết được $N$. Tại sao? Nếu bài báo được lựa chọn có tính *thời sự* thấp, thì tính *tin cậy* sẽ cao, và ngược lại. 

### 6.3.1 Nỗi buồn collider

Đặt câu hỏi độ tuổi sẽ ảnh hưởng như thế nào đến niềm vui, nếu ta khảo sát độ tuổi và niềm vui của rất nhiều người. Nếu có tương quan, vậy nó có phải nhân quả không? Giả sử, khi con người được sinh ra đã có một giá trị niềm vui nền, và nó thay đổi theo độ tuổi. Tuy nhiên, niềm vui cũng ảnh hưởng đến nhiều sự kiện trong cuộc sống, ví dụ như hôn nhân. Người vui vẻ sẽ dễ dàng thành hôn hơn. Một biến khác ảnh hưởng đến hôn nhân là độ tuổi. Sống càng lâu thì tỉ lệ kết hôn cao. Và đây là mô hình:

![](/assets/images/fig 6-9.png)

Niềm vui $H$ và độ tuổi $A$ cùng gây ra kết hôn $M$. Cho nên $M$ là một collider. Mặc dù không có quan hệ nhân quả nào giữa niềm vui và độ tuổi, nhưng nế chúng ta hồi quy thêm biến $M$ vào, nó sẽ tạo ra tương quan thống kê giữa độ tuổi và niềm vui. Và ta sẽ lầm tưởng rằng niềm vui thay đỏi theo độ tuổi, mà trên thực tế nó là hằng định.

Ta sẽ mô phỏng lại những gì ta đã nói ở trên. Thiết kế mô phỏng như sau:
1. Mỗi năm, 20 người ra đời với giá trị niềm vui phân phối theo Uniform.
2. Mỗi năm, mọi người sẽ thêm 1 tuổi. Niềm vui không thay đổi.
3. Ở tuổi 18, cá nhân co thể kết hôn. Tỉ lệ kết hôn tương đương với niềm vui cá nhân.
4. Khi kết hôn, người đó luôn giữ trạng thái kết hôn.
5. Sau 65 tuổi, người đó sẽ rời khỏi mẫu.

```python
def sim_happiness(seed=1977, N_years=1000, max_age=65, N_births=20, aom=18):
    # age existing individuals & newborns
    A = jnp.repeat(jnp.arange(1, N_years + 1), N_births)
    # sim happiness trait - never changes
    H = jnp.repeat(jnp.linspace(-2, 2, N_births)[None, :], N_years, 0).reshape(-1)
    # not yet married
    M = jnp.zeros(N_years * N_births, dtype=jnp.uint8)

    def update_M(i, M):
        # for each person over 17, chance get married
        married = dist.Bernoulli(logits=(H - 4)).sample(random.PRNGKey(seed + i))
        return jnp.where((A >= i) & (M == 0), married, M)

    M = lax.fori_loop(aom, max_age + 1, update_M, M)
    # mortality
    deaths = A > max_age
    A = A[~deaths]
    H = H[~deaths]
    M = M[~deaths]

    d = pd.DataFrame({"age": A, "married": M, "happiness": H})
    return d


d = sim_happiness(seed=1977, N_years=1000)
print_summary(dict(zip(d.columns, d.T.values)), 0.89, False)
```

|           |  mean |   std | median |  5.5% | 94.5% |  n_eff | r_hat |
|       age | 33.00 | 18.77 |  33.00 |  1.00 | 58.00 |   2.51 |  2.64 |
| happiness |  0.00 |  1.21 |   0.00 | -2.00 |  1.58 | 338.78 |  1.00 |
|   married |  0.28 |  0.45 |   0.00 |  0.00 |  1.00 |  48.04 |  1.18 |

![](/assets/images/fig 6-10.png)

Mô phỏng này sẽ chạy 1000 năm, kết quả thu được là 1300 mẫu quan sát, đúng với DAG đã mô tả ở trên. Giả sử bạn gặp data này nhưng không biết DAG đứng sau là như thế nào, bạn đặt câu hỏi độ tuổi sẽ ảnh hưởng niềm vui ra sao. Bạn lý luận rằng, tình trạng hôn nhân là một biến ảnh hưởng đến niềm vui. Mô hình tuyến tính của bạn như sau:

$$ \mu_i =\alpha_{\text{mid}[i]} + \beta_A A_i $$

Trong đó `mid` là tình trạng hôn nhân. Dùng prior cho intercept dưới dạng *index* sẽ dễ dàng hơn cho kiểu dữ liệu phân nhóm. Về prior, ta xem xét slope $\beta_A$ trước, bởi vì intercept được suy diễn dựa vào slope. Ta có thể scale biến $A$ trước.

```python
d2 = d[d.age > 17].copy()  # only adults
d2["A"] = (d2.age - 18) / (65 - 18)
```

Biến $A$ mới sẽ có giá trị từ 0 đến 1. Niềm vui sẽ ở thang điểm khác, từ -2 đến +2 trong data này. Nhớ rẳng 95% diện tích xác suất của data nằm ở khoảng -2SD đến 2SD, cho nên alpha nên có prior là Normal(0, 1). Ta sẽ tính posterior, cùng với đặt index cho biến $M$.

```python
d2["mid"] = d2.married

def model(mid, A, happiness):
    a = numpyro.sample("a", dist.Normal(0, 1).expand([len(set(mid))]))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a[mid] + bA * A
    numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)

m6_9 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_9,
    optim.Adam(1),
    Trace_ELBO(),
    mid=d2.mid.values,
    A=d2.A.values,
    happiness=d2.happiness.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_9 = svi.get_params(state)
post = m6_9.sample_posterior(random.PRNGKey(1), p6_9, (1000,))
print_summary(post, 0.89, False)
```

|       |  mean |  std | median |  5.5% | 94.5% |   n_eff | r_hat |
|  a[0] | -0.20 | 0.06 |  -0.20 | -0.30 | -0.10 | 1049.96 |  1.00 |
|  a[1] |  1.23 | 0.09 |   1.23 |  1.09 |  1.37 |  898.97 |  1.00 |
|    bA | -0.69 | 0.11 |  -0.69 | -0.88 | -0.53 | 1126.51 |  1.00 |
| sigma |  1.02 | 0.02 |   1.02 |  0.98 |  1.05 |  966.00 |  1.00 |

Mô hình khá khẳng định rằng tuổi tương quan âm với niềm vui. Ta sẽ so sánh suy luận từ mô hình này mà không có tình trạng hôn nhân.

```python
def model(A, happiness):
    a = numpyro.sample("a", dist.Normal(0, 1))
    bA = numpyro.sample("bA", dist.Normal(0, 2))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bA * A
    numpyro.sample("happiness", dist.Normal(mu, sigma), obs=happiness)

m6_10 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m6_10, optim.Adam(1), Trace_ELBO(), A=d2.A.values, happiness=d2.happiness.values
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_10 = svi.get_params(state)
post = m6_10.sample_posterior(random.PRNGKey(1), p6_10, (1000,))
print_summary(post, 0.89, False)
```

|       |  mean |  std | median |  5.5% | 94.5% |  n_eff | r_hat |
|     a |  0.01 | 0.08 |   0.01 | -0.12 |  0.12 | 931.50 |  1.00 |
|    bA | -0.01 | 0.13 |  -0.01 | -0.22 |  0.21 | 940.88 |  1.00 |
| sigma |  1.21 | 0.03 |   1.21 |  1.17 |  1.26 | 949.78 |  1.00 |

Mô hình thứ 2 thì ngược lại, không có quan hệ giữa độ tuổi và niềm vui.

Hiện tượng này là những gì ta sẽ gặp nếu đặt điều kiện lên biến collider. Collider ở đây là biến $M$, tình trạng hôn nhân. Khi chúng ta biết ai đó là kết hôn hay chưa, biết thêm độ tuổi không cung cấp thêm thông tin nào khác. Mô hình đầu tiên cho mối tương quan thống kê, không phải tương quan nhân quả.

Bạn có thể thấy được hiện tượng này ở hình trên. Nhìn vào các điểm xanh, nhữngn người đã kết hôn. Trong các điểm xanh, người lớn tuổi thì ít niềm vui hơn. Bởi vì theo thời gian nhiều người kết hôn hơn, và trung bình của niềm vui sẽ tiệm cận với niềm vui trung bình quần thể. Nhìn vào các điểm trắng, bởi vì người có giá trị niềm vui cao hơn đi qua phân điểm xanh, cho nên có mối tương quan âm giữa tuổi và niềm vui ở cả 2 quần thể.

### 6.3.2 DAG bị ám

Collider bias xuất phát từ việc đặt điều kiện lên một hậu quả chung, như ví dụ trước. Nếu có thể dựng sơ đồ nhân quả, chúng ta có thể tránh được điều này. Nhưng việc phát hiện collider không dễ dàng chút nào, bởi có nhiều nguồn không đo đạc được. Nguồn không đo đạc được vẫn gây ra collider bias. Có thể nói DAG của chúng ta đã bị ám.

Giả sử ta muốn suy luận ảnh hưởng của cả cha mẹ $(P)$ và ông bà $(G)$ lên kết quả giáo dục của con cái $(C)$. Bởi vì ông bà ảnh hưởng đến giáo dục con cái, nên có mũi tên từ $G \to P$.

![](/assets/images/fig 6-11.png)

Nhưng giả sử có thêm một yếu tố không đo đạc được, ảnh hưởng đến cả cha mẹ lẫn con cái, như yếu tố môi trường, nhưng lại không ảnh hưởng đến ông bà ( có thể nhà xa ). Lúc đó DAG của chúng ta bị ám bởi biến $U$ không được quan sát:

![](/assets/images/fig 6-12.png)

Bây giờ $P$ là kết quả chung của $G$ và $U$, cho nên nếu ta đặt điều kiện lên $P$, suy luận $G \to C$ sẽ bị bias, cho dù có hay không đo đạc $U$. Ta sẽ mô phỏng thử 300 cặp ba ông bà, cha mẹ, và con cái.

```python
N = 200  # number of grandparent-parent-child triads
b_GP = 1  # direct effect of G on P
b_GC = 0  # direct effect of G on C
b_PC = 1  # direct effect of P on C
b_U = 2  # direct effect of U on P and C

with numpyro.handlers.seed(rng_seed=1):
    U = 2 * numpyro.sample("U", dist.Bernoulli(0.5).expand([N])) - 1
    G = numpyro.sample("G", dist.Normal().expand([N]))
    P = numpyro.sample("P", dist.Normal(b_GP * G + b_U * U))
    C = numpyro.sample("C", dist.Normal(b_PC * P + b_GC * G + b_U * U))
    d = pd.DataFrame({"C": C, "P": P, "G": G, "U": U})
```

Để ý rằng, slope $b_GC =0$, ông bà không ảnh hưởng gì đến con cái, mục đích để làm rõ hơn hiệu ứng bias do collider. Tương tự, biến $U$ là biến nhị phân $(-1,1)$.

Bây giờ chúng ta sẽ suy luận ảnh hưởng của ông bà. Bởi vì hiệu ứng của ông bà có liên quan gián tiếp đến con cái thông qua cha mẹ, nên ta phải thêm biến $P$.

```python
def model(P, G, C):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b_PC * P + b_GC * G
    numpyro.sample("C", dist.Normal(mu, sigma), obs=C)


m6_11 = AutoLaplaceApproximation(model)
svi = SVI(
    model, m6_11, optim.Adam(0.3), Trace_ELBO(), P=d.P.values, G=d.G.values, C=d.C.values
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_11 = svi.get_params(state)
post = m6_11.sample_posterior(random.PRNGKey(1), p6_11, (1000,))
print_summary(post, 0.89, False)
```

|       |  mean |  std | median |  5.5% | 94.5% |   n_eff | r_hat |
|     a | -0.08 | 0.10 |  -0.09 | -0.24 |  0.06 | 1049.96 |  1.00 |
|  b_GC | -0.71 | 0.11 |  -0.71 | -0.89 | -0.55 |  813.76 |  1.00 |
|  b_PC |  1.72 | 0.04 |   1.72 |  1.65 |  1.79 |  982.64 |  1.00 |
| sigma |  1.39 | 0.07 |   1.39 |  1.28 |  1.49 |  968.54 |  1.00 |

Hiệu ứng của cha mẹ khá cao, lớn gấp 2 lần hơn giá trị mô phỏng của nó. Không có gì ngạc nhiên, bởi vì tương quan giữa $P$ và $C$ còn do $U$, và mô hình thì không biết $U$. Ngạc nhiên hơn là hiệu ứng trực tiếp của ông bà là âm. Mô hình tuyến tính không sai, mà là diễn giải nhân quả có vấn đề.

![](/assets/images/fig 6-13.png)

Nhìn hình trên, trục hoành là giáo dục của ông bà, trục tung là giáo dục của con cái, có 2 nhóm điểm. Điểm xanh là con cái ở môi trường tốt ($U=1$). Điểm đen là con cái ở môi trường xấu ($U=-1$). Nhìn tổng thể thì $G$ có hiệu ứng tích cực lên $P$, nhưng toàn bộ hiệu ứng này là từ cha mẹ. Tại sao? Bởi vì data này theo mô phỏng của chúng ta. Hiệu ứng của $G$ là zero.

Vậy tương quan âm từ đâu, khi chúng ta đặt điều kiện lên cha mẹ? Đặt điều kiện lên cha mẹ giống như chọn nhóm cha mẹ giống nhau về giáo dục. Trong hình trên, những cha mẹ cùng giáo dục trong khoảng 45th đến 60th percentile được tô đậm. Nếu ta vẽ dường hồi quy lên bằng những điểm này, hồi quy $C$ trên $G$, slope sẽ là số âm. Tại sao như vậy?

Bởi vì khi ta biết $P$, biết thêm $G$ sẽ gián tiếp cho ta biết môi trường giáo dục $U$, và $U$ ảnh hưởng $C$. Biến $U$ không đo đạc được làm cho $P$ thành collider, và đặt điều kiện lên $P$ tạo ra bias. Ta phải làm gì? Ta phải đo $U$. Đây là mô hình hồi quy tuyến tính có $U$:

```python
def model(P, G, U, C):
    a = numpyro.sample("a", dist.Normal(0, 1))
    b_PC = numpyro.sample("b_PC", dist.Normal(0, 1))
    b_GC = numpyro.sample("b_GC", dist.Normal(0, 1))
    b_U = numpyro.sample("U", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b_PC * P + b_GC * G + b_U * U
    numpyro.sample("C", dist.Normal(mu, sigma), obs=C)


m6_12 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m6_12,
    optim.Adam(1),
    Trace_ELBO(),
    P=d.P.values,
    G=d.G.values,
    U=d.U.values,
    C=d.C.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(2000))
p6_12 = svi.get_params(state)
post = m6_12.sample_posterior(random.PRNGKey(1), p6_12, (1000,))
print_summary(post, 0.89, False)
```

|       |  mean |  std | median |  5.5% | 94.5% |   n_eff | r_hat |
|     U |  1.89 | 0.17 |   1.90 |  1.61 |  2.14 | 1009.20 |  1.00 |
|     a | -0.03 | 0.08 |  -0.03 | -0.15 |  0.10 |  765.90 |  1.00 |
|  b_GC |  0.03 | 0.10 |   0.04 | -0.13 |  0.20 | 1032.11 |  1.00 |
|  b_PC |  1.02 | 0.07 |   1.02 |  0.90 |  1.14 | 1106.88 |  1.00 |
| sigma |  1.09 | 0.05 |   1.09 |  0.99 |  1.17 |  832.71 |  1.00 |

Và slope của posterior này phù hợp với data mà chúng ta mô phỏng.

>Paradox thống kê và giải thích nhân quả: Ví dụ này là một ví dụ điển hình của Simpson's paradox. Việc thêm biến mới làm đảo dấu của tương quan giữa các các biến dự đoán với biến kết cục. Thông thường, Simpson's paradox được trình bày dưới dạng thêm biến mới là co ích, nhưng trong trường hợp này, nó gây hiểu lầm cho người làm thống kê. Simpson's paradox là hiện tượng thống kê. Để biết sự đảo dấu này là có quan hệ nhân quả chính xác, ta cần thêm công cụ nhiều hơn đơn thuần mô hình thống kê.

## <center>6.4 Đối phó với sai lệch (confounding)</center><a name="4"></a>

Trong chương này và chương trước, có nhiều ví dụ về cách chúng ta dùng hồi quy đa biến để đối phó với sai lệch. Và chúng ta cũng thấy hồi quy đa biến cũng có thể gây sai lệch - thêm sai biến sẽ gây ảnh hưởng suy luận. Hi vọng rằng bạn đã biết sợ hãi việc thêm một biến, cho hệ thống hồi quy tự chạy. Và bạn có niềm tin rằng suy luận đúng nếu chúng ta cẩn thận và tự trang bị đủ kiến thức.

Nhưng nguyên tắc nào giải thích cho việc thêm hoặc bỏ các biến ảnh hưởng suy luận?

Xem xét định nghĩa sai lệch trong trường hợp suy luận hiệu ứng từ biến dự đoán $X$ đến biến kết cục $Y$. Ví dụ tương quan giữa đào tạo $E$ và bậc lương $W$. Vấn đề là có rất nhiều biến không được quan sát $U$ ảnh hưởng cả $E$ và $W$, như môi trường, giai đình, bạn bè.

![](/assets/images/fig 6-14.png)

Nếu ta hồi quy $W$ bằng $E$, ước lượng hiệu ứng nhân quả sẽ bị sai lệch bởi $U$. Nó bị sai lệch, bởi vì có 2 con đường nối giữa $E$ và $W$.
1. $E \to W$
2. $E \gets U \to W$.

"Con đường" là dãy các biến số nối giữa biến dự đoán $X$ và biến kết cục $Y$, bỏ qua chiều của các mũi tên. Cả 2 con đường này nếu tạo tương quan thống kê giữa $E$ và $W$. Nhưng chỉ có con đường đầu tiên là nhân quả, con đường thứ hai là không nhân quả. Tại sao? Bởi vì nếu con đường thứ 2 tồn tại, thay đổi $E$ sẽ không ảnh hưởng $W$. Toàn bộ hiệu ứng nhân quả từ $E \to W$ ở con đường thứ nhất.

Làm sao để cách ly con đường nhân quả. Giải pháp nổi tiếng nhất là chạy nghiên cứu can thiệp. Nếu chúng ta có thể gán giáo dục ngẫu nhiên, nó thay đổi DAG:

![](/assets/images/fig 6-15.png)

Kiểm soát loại bỏ ảnh hưởng của $U$ vào $E$. Biến không được quan sát không ảnh hưởng đến giáo dục khi chúng ta tự quyết định giáo dục. Với sự loại bỏ ảnh hưởng từ $U$ lên $E$, con đường $E \gets U \to W$ bị mất đi. Nó chặn con đường thứ hai. Khi con đường bị chặn, chỉ còn một con đường cho thông tin đi từ $E$ đến $W$, và đo đạc hiệu ứng giữa $E$ và $W$ sẽ cho một ước lượng tốt cho suy luận nhân quả. Kiểm soát sẽ loại trừ sai lệch, bởi vì nó chặn những con đường khác từ $E$ sang $W$.

May mắn thay, có phương pháp thống kê học để đạt được điều này, mà không cần kiểm soát $E$. Cách rõ ràng nhất là thêm $U$ vào mô hình, đặt điều kiện vào $U$. Tại sao điều này lại loại bỏ được sai lệch? Bởi vì nó chặn dòng chảy thông tin từ $E$ tới $W$ qua $U$. Nó chặn con đường thứ hai.

Để hiểu tại sao đặt điều kiện lên $U$ chặn con đường $ E \gets U \to W$, bạn cần nghĩ con đường này là một mô hình độc lập khác. Khi bạn biết $U$, biết thêm $E$ không cho thông tin gì thêm về $W$. Giả sử $U$ là GPD trung bình tại một vùng. Vùng giàu có hơn có nhiều trường hơn, dẫn đến giáo dục tốt hơn, cũng như công việc có lương $W$ khá hơn. Nếu bạn không biết vùng mà người đó sống, biết được giáo dục $E$ của người đó sẽ cho thêm thông tin về mức lương $W$, bởi vì $E$ và $W$ đều tương quan với vùng miền sinh sống. Nhưng khi bạn biết được vùng sinh sống, với điều kiện không còn đường nào khác giữa $E$ và $W$, biết $E$ sẽ không cho thông tin thêm về $W$. 

### 6.4.1. Chặn backdoor

Chặn các con đường gây sai lệch giữa biến dự đoán $X$ và biến kết cục $Y$ còn gọi là chặn backdoor. Chúng ta không muốn có quan hệ ảo nào trong những con đường không phải nhân quả mà đi vào $X$. Trong ví dụ trên, $E \gets U \to W$ là backdoor, bởi nó vào $E$ và kết nối $E$ với $W$. Con đường này là không mang tính nhân quả, nhưng vẫn tạo tương quan giữa $E$ và $W$.

Có một tin tốt là, với sơ đồ nhân quả DAG, có thể cho ta biết ta bên kiểm soát biến nào để chặn backdoor. Nó cũng cho ta biết không nên chặn biến nào, để tránh tạo ra sai lệch mới. Và tin tốt hơn nữa, chỉ có 4 loại quan hệ của biến để tạo nên mọi loại DAG. Cho nên bạn chỉ cần hiểu 4 thứ và cách thông tin lan truyền với nhau như thế nào. 

![](/assets/images/fig 6-16.png)

1. Loại quan hệ đầu tiên là *phân nhánh (fork)*, $X \gets Z \to Y$, là một loại sai lệch cổ điển. Nếu đặt điều kiện lên $Z$, biến $X$ sẽ không cho thêm thông tin về $Y$. 

2. Loại quan hệ thứ hai là *ống (pipe)*, $X \to Z \to Y$. Tương tụ như *fork*, điều kiện lên $Z$ sẽ chặn con đường này. Bạn gặp quan hệ này ở ví dụ trông cây và nấm.

3. Loại quan hệ thứ ba là *xung đột (collider)*, $X \to Z \gets Y$. Ngược lại với hai loại trên, điều kiện lên $Z$ sẽ mở con đường này, như ở ví dụ trên. Khi con đường mở, thông tin sẽ chạy từ $X \to Y$. Nhưng thực tế là, $X$ và $Y$ không có quan hệ nhân quả.

4. Loại quan hệ thứ tư là *con cháu (descendent)*. Nó là biến bị ảnh hưởng bởi biến khác. Điều kiện lên biến con cháu giống như đặt điều kiện một phần lên cha của nó. Trong hình trên, đặt điều kiện lên $D$ sẽ đặt điều kiện lên $Z$, nhưng ở mức độ nhẹ hơn. Và $Z$ là collider lên mở con đường từ $X \to Y$. Tính chất của biến con cháu sẽ phụ thuộc vào biến cha. Tình huống gặp rất nhiều trên thực tế, vì đôi khi ta không thể đo đạc trực tiếp một hiện tượng nào đó và thông qua dụng cụ gián tiếp.

Cho dù DAG có phức tạp cỡ nào, nó luôn dựa trên 4 quan hệ kể trên. Và bạn đã biết cách đóng và mở các loại quan hệ, bạn (hoặc máy tính) có thể tìm ra biến nào nên thêm vào hoặc loại ra. Sau đây là công thức:

1. Liệt kê toàn bộ con đường nối $X$ và $Y$. 

2. Xác định những con đường đó là đóng hay mở. Một con đường là mở trừ phi nó chứa collider.

3. Xác định những con đường đó là backdoor hay không. Backdoor sẽ có mũi tên đi vào $X$.

4. Nếu có con đường backdoor nào đang mở, ta quyết định đặt điều kiện lên nó để chặn con đường.

Hãy xem các ví dụ sau.

### 6.4.2 Hai con đường

DAG dưới đây chứa biến dự đoán là $X$, biến kết cục là $Y$, biến không quan sát được là $U$, 3 biến quan sát được còn lại là $A$, $B$, $C$.

![](/assets/images/fig 6-17.png)

Ta quan tâm đến con đường $X \to Y$,  và quan hệ nhân quả của nó. Vậy ta chọn biến nào để vào mô hình, để ước lượng đúng quan hệ nhân quả? Có 2 backdoor:

1. $X \gets U \gets A \to C to Y$

2. $X \gets U \to B \gets C \to Y$

Cả hai đều có thể gây sai lệch cho suy luận nhân quả. Bây giờ tìm con đường nào là đang mở. Nếu backdoor đang mở, ta phải đóng nó. Nếu backdoor đang đóng, ta tránh mở nó để tạo ra sai lệch.

Con đường đầu tiên đi qua $A$, đường này đang mở, bởi vì không có collider nào trong đó. Trong đó có một *fork* và hai *pipe*. Thông tin sẽ đi qua con đường này, tạo ra sai lệch nhân quả $X \to Y$. Nó là một backdoor, để đóng backdoor này, ta phải thêm đặt điều kiện lên một biến trên con đường này. Ta không thể đặt điều kiện lên $U$, bởi vì nó là không quan sát được, chỉ còn $A$ và $C$. Cả hai đều có thể chặn backdoor. Bạn có thể dùng phần mềm `daggity` để phân tích DAG này.

Đặt điều kiện lên $A$ và $C$ đều được, nhưng $C$ sẽ cho kết quả suy luận nhân quả chính xác hơn.

Giờ ta xem con đường thứ hai, backdoor này chứ collider, $U \to B \gets C$. Cho nên nó đã đóng. Thực vậy, nếu bạn đặt điều kiện lên B, nó sẽ mở con đường này, tạo sai lệch. Hiệu ứng nhân quả từ $X \to Y$ sẽ thay đổi, mà nếu bạn không có DAG, bạn sẽ không biết thay đổi này là giúp đỡ hay gây sai lệch. Cho nên, nếu cho thêm một biến vào mô hình hồi quy, mà hệ số hồi quy thay đổi, không có nghĩa là mô hình tốt lên, có thể bạn đã gặp collider. 

### 6.4.3. Backdoor waffles.

Ví dụ cuối cùng với tương quan giữa Waffles House và tỉ lệ ly hôn ở Chương 5. Ta sẽ dựng một DAG, dùng nó để tìm tập tối thiểu các biến số để suy luận nhân quả.

![](/assets/images/fig 6-18.png)

Trong sơ đồ này, $S$ là bang đó có nằm ở phía Nam hay không, $A$ là tuổi kết hôn trung bình, $M$ là tỉ lệ kết hôn, $W$ là số cửa hàng Waffl House, $D$ là tỉ lệ ly dị. Sơ đồ này cho rằng các bang phía Nam sẽ có tuổi kết hôn nhỏ hơn ($S \to A$), tỉ lệ kết hôn cao hơn trực tiếp ($S \to M$) và gián tiếp ($S \to A \to M$), và cũng có nhiều cửa hàng hơn ($S \to W$). Tuổi và tỉ lệ kết hôn ảnh hưởng lên ly dị.

Có 3 backdoor từ $W$ đến $D$. Tất cả đều qua $S$, cho nên ta có thể đóng tất cả backdoor bằng $S$. Có thể xác nhận lại bằng `daggity` [link](http://www.dagitty.net/dags.html).

Model code:

```
dag {
A [pos="-0.890,-0.341"]
D [outcome,pos="-0.481,-0.334"]
M [pos="-0.685,-0.498"]
S [pos="-0.901,-0.652"]
W [exposure,pos="-0.487,-0.643"]
A -> D
A -> M
M -> D
S -> A
S -> M
S -> W
W -> D
}
```

![](/assets/images/fig 6-19.png)

Vậy ta có thể kiểm soát $(A, M)$ hoặc chỉ $S$.

DAG này đương nhiên là không thoả mãn - nó không giả định sai lệch không quan sát được, mặc dù với tập data này thì ít xảy ra. Nhưng ta vẫn học được điều gì đó bằng phân tích. Trong khi data không thể nói được DAG nào là đúng, nhưng nó có thể nói được DAG nào là sai. Phần trước, ta thảo luận mối quan hệ độc lập có điều kiện (conditional independency), hay còn gọi là suy luận kiểm tra được (testable implication). Chúng là những cặp biến không quan hệ với nhau, khi được đặt điều kiện lên tập hợp biến khác. Bằng xem xét những mối quan hệ độc lập có điều kiện này, ta ít ra kiểm tra được những đặc trưng của DAG.

Giờ bạn biết được những sai lệch cơ bản, bạn có thể tự diễn giải kết quả quan hệ độc lập có điều kiện. Trong ví dụ này có 3 cặp quan hệ độc lập có điều kiện (hình trên).

Dòng đầu tiên là "tuổi trung bình kết hôn thì độc lập với số lượng cửa hàng Waffle House, điều kiện là bang đó ở phía Nam." Dòng thứ hai, ly dị và phía Nam độc lập với nhau nếu đồng thời đặt điều kiện lên tuổi trung bình kết hôn, tỉ lệ kết hôn và số lượng cửa hàng. Cuối cùng, tỉ lệ kết hôn và số lượng cửa hàng là độc lập, khi điều kiện là ở phía Nam.

Bạn sẽ phải dựng mô hình cho từng mối quan hệ trên, và kiểm tra tính phù hợp của nó. Nếu có sai, bạn phải chỉnh sửa lại DAG, thêm hoặc xoá các mũi tên, giới thiệu biến mới,..

>**DAG thôi vẫn không đủ.** Nếu bạn không có mô hình thực của hệ thống, DAG rất tuyệt vời. Nó làm cho giả định rõ ràng, và dễ dàng để đánh giá. Và nếu có gì khác, nó chỉ rõ mối nguy hiểm tiềm tàng khi dùng hồi quy tuyến tính thay vì cho giả thuyết. Nhưng DAG không phải đích đến cuối cùng. Khi bạn biết được mô hình động của hệ thống, bạn không cần DAG. Thực vậy, rất nhiều mô hình động có hành vi phức tạp, nhạy cảm với những giả định ban đầu, không thể trình bày bằng DAG đơn thuần. Nhưng mô hình vẫn có thể được phân tích và can thiệp nhân quả. Thực vậy, mô hình nhân quả theo cấu trúc chuyên ngành có thể suy luận nhân quả hiệu quả hơn so với DAG cùng cấu trúc. Càng nhiều giả định chính xác, suy luận càng mạnh hơn, power cao hơn.  
>Sự thật DAG không được dùng cho mọi thứ là khỏi tranh cãi. Mọi công cụ giả thuyết đều có giới hạn. Nhưng DAG là một công cụ tốt để trình bày suy luận nhân quả cho người mới học.

---

Phép tính `do`. Để định nghĩa chính xác sai lệch, ta cần phép dùng ký hiệu. Sai lệch xảy ra khi:

$ Pr ( Y \|X) = Pr (Y \| do(X) ) $

Dấu $do(X)$ nghĩa là chặn tất cả các backdoor đến $X$, nhưng ta đã thí nghiệm. Phép tính `do` thay đổi sơ đồ nhân quả, chặn các backdoor. Nó định nghĩa mối quan hệ nhân quả, bởi vì $Pr(Y \| do(X))$ cho ta biết kết quả $Y$ khi kiểm soát $X$, với một DAG. Ta có thể nói rằng $X$ là nguyên nhân của $Y$ khi  $Pr(Y \| do(X)) \neq Pr(Y \| do(not-X))$. Còn $Pr(Y \| X) \neq Pr(Y \| not-X)$ không có chặn các backdoor. Chú ý rằng `do` không cho hiệu ứng nhân quả trực tiếp, mà là toàn bộ hiệu ứng. Để rút hiệu ứng trực tiếp, ta cần chặn nhiều backdoor hơn. `do` giúp ta tạo ra nhiều kỹ thuật suy luận nhân quả, ngay cả khi có backdoor không dóng được. Ta sẽ gặp lại nó ở chương sau.

---

*Chapter end*