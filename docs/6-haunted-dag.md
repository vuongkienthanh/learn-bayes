---
title: "Chapter 6: The Haunted DAG & The Causal Terror"
description: "Chương 6: DAG bị ám và sự kinh hoàng của nhân quả"
---

- [6.1 Hiện tượng đa cộng tuyến](#a1)
- [6.2 Sai lệch hậu điều trị](#a2)
- [6.3 Sai lệch xung đột](#a3)
- [6.4 Đối phó với nhiễu](#a4)
- [6.4 Tổng kết](#a5)

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

## <center>6.1 Hiện tượng đa cộng tuyến</center><a name="a1"></a>

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

## <center>6.2 Sai lệch hậu điều trị</center><a name="a2"></a>

Thông thường chúng ta quan tâm đến các lỗi sai trong suy luận xuất phát từ việc không đủ biến dự đoán. Những lỗi sai này thường được gọi là **SAI LỆCH THIẾU BIẾN SỐ (OMITTED VARIABLE BIAS)**, và các ví dụ ở chương trước đã minh hoạ nó. Ít được lo lắng hơn trong các lỗi sai suy luận xuất phát từ việc *thêm biến*. Nhưng **SAI LỆCH DƯ BIẾN SỐ (INCLUDED VARIABLE BIAS)** là có thật. Thí nghiệm ngẫu nhiên cẩn thận cỡ nào cũng có thể bị huỷ hoại dẽ dàng như nghiên cứu quan sát không kiểm soát. Thêm biến một cách mù quáng vào salad nhân quả không bao giờ là ý tưởng tốt.

Sai lệch dư biến số có rất nhiều loại. Đầu tiên là **SAI LỆCH HẬU ĐIỀU TRỊ (POST-TREATMENT BIAS)**. Sai lệch hậu điều trị là một nguy cơ có trong mọi loại nghiên cứu. Từ "hậu điều trị" có từ việc thiết kế thí nghiệm. Ví dụ bạn đang trồng cây tại một nhà kính. Bạn muốn biết sự khác nhau về phát triển giữa nhiều loại điều trị phân bón kháng nấm khác nhau trên cây, bởi vì nấm trên cây thường cản trở sự tăng trưởng của cây. Lúc bắt đầu, cây được giao hạt và nảy mầm, thu thập dữ liệu chiều cao ban đầu của cây. Các điều trị phân bón kháng nấm khác nhau được ghi nhận. Số đo cuối cùng là chiều cao của cây và sự hiện diện của nấm. Có bốn biến số được quan tâm: chiều cao ban đầu, chiều cao cuối cùng, điều trị, và hiện diện của nấm.

Chiều cao cuối cùng là kết cục quan tâm. Nhưng biến nào cần được cho vào mô hình? Nếu mục tiêu là suy luận nhân quả về hiệu quả điều trị, bạn không nên bao gồm biến nấm. bởi vì nó là *hiệu ứng hậu điều trị*.

Hãy mô phỏng data, để làm ví dụ này rõ ràng hơn và xem cái gì xảy ra khi chúng ta cho thêm biến hậu điều trị.

<b>code 6.13</b>
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
<samp>            mean   std  median   5.5%  94.5%  n_eff  r_hat
   fungus   0.31  0.46    0.00   0.00   1.00  18.52   1.17
       h0   9.73  1.95    9.63   7.05  13.33  80.22   0.99
       h1  13.72  2.47   13.60  10.73  18.38  43.44   1.08
treatment   0.50  0.50    0.50   0.00   1.00   2.64    inf</samp>

Bây giờ bạn có một DataFrame `d` với data thí nghiệm cây được mô phỏng.

<div class="alert alert-info">
<p><strong>Học suy luận nhân quả.</strong> Nguy hiểm của sai lệch hậu điều trị được khám phá từ lâu. Nhiều nhà khoa học được dạy rằng khi đặt điều kiện trên biến hậu điều trị là có nhiều nguy cơ, còn biến tiền điều trị thì an toàn. Quan điểm này có thể dẫn đến ước lượng hợp lý trong nhiều trường hợp. Nhưng nó không phải nguyên tắc chung. Biến tiền điều trị cũng có thể gây sai lệch, và bạn sẽ thấy nó sau trong chương này. Không có gì sai, theo nguyên tắc, trong quan điểm đó. Chúng là an toàn trong bối cảnh chúng được chế tạo. Nhưng chúng ta vẫn cần một nguyên tắc chung để biết được khi nào dùng đến nó.</p></div>

### 6.2.1 Thiết kế prior

Khi thiết kế mô hình, tốt hơn là bạn giả bộ không biết quy trình xử lý tạo data như trên. Trong nghiên cứu thực, bạn sẽ không biết được quy trình xử lý tạo ra data thực. Nhưng bạn sẽ có rất nhiều kiến thức khoa học để hướng dẫn xây dựng mô hình. Cho nên hãy bỏ ra ít thời gian để nghiêm túc thực hiện phân tích thử này.

Chúng ta biết rằng chiều cao tại thời điểm $t=1$ sẽ cao hơn so với thời điểm $t=0$, cho dù chúng được đo lường ở thang nào. Cho nên chúng ta đặt tham số trên thang đo là tỉ lệ so với chiều cao vào lúc $t=0$, hơn là thang đo tuyệt đối của data, để chúng ta có thể đặt prior dễ hơn. Để đơn giản, hãy tập trung vào chỉ biến chiều cao, bỏ qua biến dự đoán. Chúng ta có mô hình tuyến tính như sau:

$$\begin{aligned}
h_{1,i} & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i  &= h_{0,i} \times p \\
\end{aligned}$$

Trong đó $h_{0,i} là chiều cao chiều cây $i$ vào thời điểm $t=0$, $h_{1,i}$ là chiều cao của nó vào thời điểm $t=1$, và $p$ là tham số đo lường tỉ lệ giữa $h_{0,i}$ so với $h_{1,i}$. Cụ thể hơn, $p=h_{1,i}/h_{0,i}$. Nếu $p=1$, tức là câu không hề thay đổi từ lúc $t=0$ đến $t=1$. Nếu $p=2$, tức là chiều cao tăng gấp đôi. Vậy nếu chúng ta đặt prior của $p$ xung quanh $p=1$, nghĩa là mong đợi không có sự thay đổi chiều cao. Nhưng chúng ta cũng nên cho phép $p<1$, trong trường hợp thí nghiệm gặp trục trặc và diệt hết các cây. Chúng ta cũng phải đảm bảo khả năng $p>0$, bởi vì nó là một tỉ lệ. Lúc ở Chương 4, chúng ta đã dùng phân phối Log-Normal, bởi vì nó luôn dương. Hãy dùng nó lần nữa. Nếu chúng ta sử dụng $p \sim \text{Log-Normal}(0,0.25)$, phân phối prior sẽ trông giống như sau:

<b>code 6.14</b>
```python
sim_p = dist.LogNormal(0, 0.25).sample(random.PRNGKey(0), (int(1e4),))
print_summary({"sim_p": sim_p}, 0.89, False)
```
<samp>       mean   std  median  5.5%  94.5%    n_eff  r_hat
sim_p  1.04  0.27    1.00  0.63   1.44  9936.32   1.00</samp>

Prior này mong đợi chiều cao bị nhỏ lại 40% hoặc cao hơn 50%. Hãy fit mô hình này để bạn có thể thấy được nó đo lường hiệu ứng tăng trưởng trung bình trong thí nghiệm này như thế nào.

<b>code 6.15</b>
```python
def model(h0, h1):
    p = numpyro.sample("p", dist.LogNormal(0, 0.25))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = h0 * p
    numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)
m6_6 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_6, optim.Adam(1), Trace_ELBO(), h0=d.h0.values, h1=d.h1.values)
p6_6, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_6.sample_posterior(random.PRNGKey(1), p6_6, (1000,))
print_summary(post, 0.89, False)
```
<samp>       mean   std  median  5.5%  94.5%    n_eff  r_hat
    p  1.39  0.02    1.39  1.36   1.42   994.30   1.00
sigma  1.84  0.13    1.84  1.65   2.06  1011.70   1.00</samp>

Chiều cao tăng trung bình khoảng 40%. Bây giờ ta thêm biến điều trị và biến hiện diện nấm vào mô hình, dựa trên mục đích đo lường hiệu ứng của điều trị và hiện diện nấm. Tham số của những biến này vẫn sẽ ở thang đo tỉ lệ. Chúng sẽ là những *thay đổi* ở tỉ lệ tăng trưởng. Vậy chúng ta sẽ dựng mô hình tuyến tính của $p$ như sau.

$$\begin{aligned}
h_{1,i} & \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i  &= h_{0,i} \times p \\
p &= \alpha +\beta_T T_i +\beta_F F_i \\
\alpha  & \sim \text{Log-Normal}(0, 0.25) \\
\beta_T & \sim \text{Normal} (0, 0.5)\\
\beta_F & \sim \text{Normal} (0, 0.5)\\
\sigma  & \sim \text{Exponential} (1) \\
\end{aligned}$$

Tỉ lệ tăng trưởng $p$ bây giờ là hàm số của các biến dự đoán. Nó trông giống như những mô hình tuyến tính khác. Prior của các slope khá là phẳng. Khoảng 95% mật độ prior nằm ở giữa -1 (giảm 100%) và +1 (tăng 100%) và hai phần ba mật độ prior nằm giữa -0.5 và +0.5. Sau khi chúng ta xong phần này, bạn có thể quay lại và thử mô phỏng với những prior này. Đây là code để ước lượng posterior.

<b>code 6.16</b>
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
p6_7, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_7.sample_posterior(random.PRNGKey(1), p6_7, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a   1.47  0.03    1.47   1.43   1.51  1049.04   1.00
   bf  -0.28  0.03   -0.28  -0.33  -0.23   910.93   1.00
   bt   0.01  0.03    0.01  -0.03   0.06  1123.06   1.00
sigma   1.39  0.10    1.39   1.21   1.54   976.96   1.00</samp>

Tham số $a$ thì giống như $p$ phần trước. Nó có posterior gần như là giống nhau. Posterior biên của $\beta_T$, hiệu ứng điều trị, là zero và khoảng tin cậy hẹp. Việc điều trị không liên quan với sự tăng trưởng. Nhưng nấm thì có vẻ gây giảm sự tăng trưởng. Cho rằng ta biết việc điều trị có ích cho sự tăng trưởng, vì vốn dĩ ta đã tự mô phỏng data như vậy, thế chuyện gì đã xảy ra?

### 6.2.2 Bị chặn bởi hệ quả

Vấn đề là `fungus` là hệ quả của `treatment`. Một cách nói khác là `fungus` là biến hậu điều trị. Khi chúng ta kiểm soát `fungus`, câu hỏi dành cho mô hình là: *Khi chúng ta biết có hay không có sự hiện diện của nấm, việc điều trị có quan trọng không?* Câu trả lời là "không", bởi vì việc điều trị ảnh hưởng sự tăng trưởng thông qua việc giảm nhiễm nấm. Nhưng cái chúng ta quan tâm ở đây, dựa vào thiết kê nghiên cứu, là tác động của điều trị lên sự tăng trưởng. Để đo lường đúngm chúng ta nên bỏ qua biến hậu điều trị `fungus`. Đây là suy luận trong trường hợp này.

<b>code 6.17</b>
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
p6_8, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_8.sample_posterior(random.PRNGKey(1), p6_8, (1000,))
print_summary(post, 0.89, False)
```
<samp>       mean   std  median  5.5%  94.5%   n_eff  r_hat
    a  1.33  0.02    1.33  1.29   1.37  930.82   1.00
   bt  0.13  0.04    0.12  0.08   0.19  880.02   1.00
sigma  1.73  0.12    1.73  1.55   1.94  948.82   1.00</samp>

Giờ thì hiệu ứng của điều trị rõ ràng là số dương, và nó đáng lẽ như vậy. Nó hợp lý hơn khi kiểm soát sự khác nhau trước điều trị, như chiều cao ban đầu `h0`, mà có thể che đậy hiệu ứng nhân quả của điều trị. Nhưng việc bao gồm biến hậu điều trị có thể che đậy luôn bản thân điều trị. Điều đó không có nghĩa là bạn không muốn cả hai biến điều trị và nấm. Sự thật là việc bao gồm `fungus` làm hệ số của `treatment` thành zero, nói lên rằng điều trị hoạt động đúng như những lý do đã lường trước. Nó nói cho chúng ta biết về cơ chế. Nhưng nếu chúng ta muốn suy luận đúng về điều trị thì phải loại bỏ biến hậu điều trị.

### 6.2.3 Nấm và sự biệt hướng

Vấn đề sẽ rõ hơn nếu dùng DAG. Trong trường hợp này, tôi sẽ cho bạn thấy cách vẽ nó bằng package `daft` trong python, và dùng `causalgraphicalmodels` để phân tích sơ đồ.

<b>code 6.18</b>
```python
plant_dag = CausalGraphicalModel(
    nodes=["H0", "H1", "F", "T"], edges=[("H0", "H1"), ("F", "H1"), ("T", "F")]
)
pgm = daft.PGM()
coordinates = {"H0": (0, 0), "T": (4, 0), "F": (3, 0), "H1": (2, 0)}
for node in plant_dag.dag.nodes:
    pgm.add_node(node, node, *coordinates[node])
for edge in plant_dag.dag.edges:
    pgm.add_edge(*edge)
with plt.rc_context({"figure.constrained_layout.use": False}):
    pgm.render()
```

![](/assets/images/dag 6-2.svg)

Vậy điều trị $T$ ảnh hưởng đến sự hiện diện của nấm $F$, sau đó ảnh hưởng đến chiều cao của cây ở thời điểm 1 là $H_1$. Chiều cao cây ở thời điểm $H_1$ cũng bị ảnh hưởng bởi chiều cao cây ở thời điểm 0, $H_0$. Nó là DAG của chúng ta. Khi chúng ta thêm biến $F$, biến hậu điều trị, vào mô hình, thì chúng ta đã chặn con đường từ $T$ đến biến kết cục. Đây là cách nói theo DAG, là việc biết thêm điều trị không nói gì thêm về kết cục, một khi chúng ta đã biết tình trạng nấm.

Một cách nói theo DAG khác, là việc đặt điều kiện trên biến $F$ đã tạo **SỰ BIỆT HƯỚNG (D-SEPARATION)**. "d" là *chiều hướng (directional)*. Sự biệt hướng nghĩa là có vài biến độc lập với nhau trong đồ thị có hướng. Không có con đường nào kết nối chúng. Trong trường hợp này, $H_1$ là biệt hướng với $T$, nhưng chỉ khi đặt điều kiện trên $F$. Đặt điều kiện trên $F$ đã chặn con đường trực tiếp $T \to F \to H_1$, làm cho $T$ và $H_1$ độc lập (biệt hướng). Trong chương trước, bạn đã thấy kí hiệu $H_1 perp\\!\\!\perp T \|F$ cho mệnh đề này, khi chúng ta thảo luận **MỐI QUAN HỆ ĐỘC LẬP CÓ ĐIỀU KIỆN**. Tại sao nó xảy ra? Không có thông tin của $T$ về $H_1$ mà không có trong $F$. Cho nên khi chúng ta biết $F$, biết thêm $T$ không cung cấp thông tin thêm về $H_1$. Bạn có thể liệt kê tất cả các quan hệ độc lập có điều kiện của DAG này:

<b>code 6.19</b>
```python
plant_dag = CausalGraphicalModel(
    nodes=["H0", "H1", "F", "T"], edges=[("H0", "H1"), ("F", "H1"), ("T", "F")]
)
plant_dag.get_all_independence_relationships()
```
<samp>[('F', 'H0', set()),
 ('F', 'H0', {'T'}),
 ('H0', 'T', set()),
 ('H0', 'T', {'F'}),
 ('H0', 'T', {'F', 'H1'}),
 ('T', 'H1', {'F'}),
 ('T', 'H1', {'F', 'H0'})]</samp>

Có 6 quan hệ độc lập ở đây. Ta tập trung vào mối quan hệ thứ 5. Nhưng những mối quan hệ khác cũng cho các cách để kiểm tra sơ đồ nhân quả. Những gì $F perp\\!\\!\perp H_0$ và $H_0 perp\\!\\!\perp T$ nói là chiều cao ban đầu, $H_0$ không liên quan vào điều trị $T$ và hiện diện của nấm $F$, giả định chúng ta không đặt điều kiện trên biến nào cả.

Điều hiển nhiên là biến hậu điều trị cũng là một vấn đề trong thiết kế nghiên cứu quan sát cũng như nghiên cứu thực nghiệm. Nhưng trong thực nghiệm, có thể dễ dàng nhận biết được biến tiền điều trị, như `h0`, và biến hậu điều trị, như `fungus`. Trong nghiên cứu quan sát thì khó nhận biết chúng hơn. Nhưng có rất nhiều cạm bẫy trong nghiên cứu thực nghiệm. Ví dụ, đặt điều kiện trên biến hậu điều trị có thể không chỉ lừa bạn nghĩa rằng điều trị không hiệu quả. Nó cũng có thể lừa bạn nghĩ rằng nó hoạt động. Xem xét DAG sau đây:

![](/assets/images/dag 6-3.svg)

Trong sơ đồ này, biến điều trị $T$ ảnh hưởng đến biến nấm $F$, nhưng nấm không ảnh hưởng tăng trưởng cây. Có thể loài cây này không hề bị ảnh hưởng có loại nấm này. Biến mới $M$ là độ ẩm (moisture). Nó ảnh hưởng cả hai $H_1$ và $F$. $M$ là dấu chấm chỉ điểm cho việc nó không được quan sát. Nguồn căn nguyên chung chưa được quan sát cho $H_1$ và $F$ nào vẫn được - dĩ nhiên không nhất thiết là độ ẩm. Hồi quy $H_1$ trên $T$ sẽ cho thấy không có quan hệ giữa điều trị và tăng trưởng. Nhưng khi cho $F$ vào mô hình, lập tức mối quan hệ này xuất hiện. Hãy thử nó. Tôi sẽ chỉ tuỳ biến mô phỏng tăng trưởng cây để cho nấm không còn ảnh hưởng lên tăng trưởng, nhưng độ ẩm $M$ ảnh hưởng cả hai $H_1$ và $F$:

<b>code 6.20</b>
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

Chạy lại mô hình trước, mô hình `m6_7`, và `m6_8`, nhưng sử dụng data trong `d2`. Bạn sẽ thấy việc thêm biến nấm sẽ gây nhiễu suy luận về hiệu quả điều trị, lần này làm cho nó trở nên nó giúp đỡ tăng tưởng, mặc dù thật ra không có hiệu ứng gì.

Kết quả này khá là bí ẩn. Tại sao $M$ là có hiệu ứng này? Phần tiếp theo là tất tần tật về những hiệu ứng như vậy.

<div class="alert alert-info">
<p><strong>Chonj lọc mô hình không có ích.</strong> Trong chương sau, bạn sẽ học cách chọn lọc mô hình bằng tiêu chuẩn thông tin. Như các phương pháp chọn lọc và so sánh mô hình khác, tiêu chuẩn này giúp tương phản và lựa chọn cấu trúc mô hình. Nhưng cách tiếp cận này không giúp ích trong ví dụ trình bày ở trên, vì mô hình bao gồm <code>fungus</code> đều fit mẫu tốt hơn và sẽ cho dự đoán ngoài mẫu tốt hơn. Mô hình <code>m6_7</code> gây hiểu sai bởi vì nó hỏi sai câu hỏi, chứ không phải vì nó sẽ cho dự đoán kém. Như đã bàn luận ở Chương 1, dự đoán và suy luận nhân quả không phải là cùng một tác vụ. Không quy trình thống kê nào có thể thay thế kiến thức khoa học và sự tập trung vào nó. Chúng ta cần nhiều mô hình bởi vì chúng giúp hiểu các con đường nhân quả, không phải chỉ để chúng ta có thể thể chọn ra một trong chúng để dự đoán.</p></div>

## <center>6.3 Sai lệch xung đột</center><a name="a3"></a>

Quay lại ví dụ đầu tiên trong chương này, tôi đã nói rằng tất cả những gì cần cho bài báo khoa học có được tương quan âm giữa tính thời sự và tính tin cậy là quá trình chọn lựa - để đăng báo - quan tâm cả hai. Bây giờ tôi muốn giải thích hiện tượng chọn lựa này có thể xảy ra trong một mô hình thống kê. Khi nó xuất hiện, nó có thể làm móp méo nghiêm trọng suy luận của chúng ta, hiện tượng được biết đến là **SAI LỆCH XUNG ĐỘT (COLLIDER BIAS)**.

Hãy xem ví dụ DAG dưới đây. Mô hình này có tính tin cậy ($T$) và tính thời sự ($N$) không liên quan với nhau trong quần thể các bài báo nộp lên hội đồng. Nhưng cả hai ảnh hưởng chọn lựa ($S$) để nhận quỹ. Đây là sơ đồ:

![](/assets/images/dag 6-4.svg)

Có 2 mũi tên cho vào $S$ nên nó là một **BIẾN XUNG ĐỘT (COLLIDER)**. Nguyên lý chính thì dễ hiểu: Khi bạn đặt điều kiện trên biến xung đột, nó tạo ra tương quan thống kê - nhưng không chắc nhân quả - giữa các nguồn căn nguyên của nó. Trong trường hợp này, khi bạn biết một bài báo được chọn ($S$), thì biết thêm tính tin cậy ($T$) sẽ cung cấp thông tin về tính thời sự ($N$) của nó. Tại sao? Bởi vì nếu, ví dụ, một bài báo được chọn có tính tin cậy thấp, thì nó phải có tính thời sự cao. Nếu không nó đã không được chọn để nhận quỹ. Chiều ngược lại cũng đúng: Nếu bài báo được lựa chọn có tính thời sự thấp, thì chúng ta suy luận nó phải có tính tin cậy cao hơn mức trung bình. Ngược lại nó sẽ không được chọn để nhận quỹ.

Đây là hiện tượng thông tin mà tạo ra tương quan giữa $T$ và $N$ trong quần thể các bài báo được chọn. và nó nghĩa là chúng ta phải chú ý đến quy trình chọn mẫu quan sát của chúng ta và có thể làm móp méo mối quan hệ giữa các biến. Nhưng cùng hiện tượng này cũng sẽ tạo ra mối quan hệ gây hiểu sai khi ở trong một mô hình thống kê, khi bạn cho thêm biến xung đột vào thành một biến dự đoán. Nếu bạn không cẩn thận, bạn có thể cho ra suy luận nhân quả sai hoàn toàn. Hãy xem ví dụ mở rộng sau đây.

### 6.3.1 Biến xung đột của nỗi buồn giả

Xem xét câu hỏi tuổi tác sẽ ảnh hưởng như thế nào đến hạnh phúc. Nếu chúng ta khảo sát rất nhiều người và đánh giá hạnh phục của họ, thì tuổi tác có liên quan với hạnh phúc không? Nếu có, thì nó có phải liên quan nhân quả không? Ở đây, tôi muốn cho bạn thấy khi kiểm soát một biến có khả năng xung đột cho hạnh phúc có thể gây suy luận sai lệch về ảnh hưởng của tuổi tác như thế nào.

Giả sử, chỉ để giảng dạy, là hạnh phúc trung bình của một người là một đặc tính được quyết định lúc mới sinh ra và không thay đổi theo tuổi. Tuy nhiên, hạnh phúc cũng ảnh hưởng đến nhiều sự kiện trong cuộc sống. Một trong những sự kiện đó là hôn nhân. Người vui vẻ sẽ dễ dàng thành hôn hơn. Một biến khác ảnh hưởng nhân quả đến hôn nhân là tuổi tác. Sống càng lâu thì xác suất kết hôn càng cao. Cho cả ba biến vào chung, và đây là mô hình nhân quả:

![](/assets/images/dag 6-5.svg)

Hạnh phúc ($H$) và tuổi tác ($A$) cùng gây ra kết hôn ($M$). Cho nên kết hôn là một biến xung đột. Mặc dù không có quan hệ nhân quả nào giữa hạnh phúc và tuổi tác, nhưng nếu chúng ta đặt điều kiện trên kết hôn - tức là, nếu chúng ta hồi quy thêm biến $M$ vào - thì nó sẽ tạo ra tương quan thống kê giữa tuổi tác và hạnh phúc. Và Điều này có thể làm chúng ta hiểu sai rằng hạnh phúc thay đổi theo tuổi tác, mà trên thực tế nó là hằng định.

Để thuyết phục bạn điều này, hãy làm một mô phỏng. Mô phỏng là hữu ích trong những ví dụ này, bởi vì đó là lúc chúng ta biết mô hình nhân quả thực sự. Nếu một quy trình không thể phát hiện ra sự thật trong ví dụ mô phỏng, thì chúng ta không nên tin nó ở thế giới thực. Lần này chúng ta sẽ làm một mô phỏng hoành tráng hơn, sử dụng mô hình theo kiểu người đại diện cho tuổi tác và hạnh phúc để tạo ra data mô phỏng dùng cho hồi quy. Thiết kế mô phỏng như sau:

1. Mỗi năm, 20 người ra đời với giá trị hạnh phúc phân phối theo Uniform.
2. Mỗi năm, mọi người sẽ thêm 1 tuổi. Hạnh phúc không thay đổi.
3. Ở tuổi 18, các cá nhân có thể kết hôn. Xác suất kết hôn mỗi năm sẽ tỉ lệ với hạnh phúc cá nhân.
4. Khi kết hôn, người đó luôn giữ trạng thái kết hôn.
5. Sau 65 tuổi, người đó sẽ rời khỏi mẫu. (Họ đi Tây Ban Nha)

Chạy thuật toán này cho 1000 năm và thu thập kết quả:

<b>code 6.21</b>
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
<samp>            mean    std  median   5.5%  94.5%   n_eff  r_hat
      age  33.00  18.77   33.00   1.00  58.00    2.51   2.64
happiness   0.00   1.21    0.00  -2.00   1.58  338.78   1.00
  married   0.28   0.45    0.00   0.00   1.00   48.04   1.18</samp>

<a name="f4"></a>![](/assets/images/fig 6-4.svg)
<details class="fig"><summary>Hình 6.4: Data mô phỏng, giả định hạnh phúc là phân phối đồng dạng và không bao giờ thay đổi. Mỗi điểm là một người. Cá nhân đã kết hôn là màu đỏ. Ở độ tuổi sau 18, người hạnh phúc nhất sẽ dễ kết hôn hơn. Tuổi càng lớn, nhiều người sẽ kết hôn hơn. Tình trạn kết hôn là một biến xung đột của tuổi tác và hạnh phúc: $A \to M \gets H$. Nếu chúng ta đặt điều kiện trên kết hôn vào hồi quy, nó sẽ làm cho chúng ta hiểu sai là hạnh phúc giảm theo tuổi tác.</summary>
{% highlight python %}plt.scatter(d[d['married']==0]['age'], d[d['married']==0]['happiness'], label='chưa kết hôn')
plt.scatter(d[d['married']==1]['age'], d[d['married']==1]['happiness'], label='đã kết hôn')
plt.gca().set(xlabel='tuổi', ylabel='hạnh phúc')
plt.legend(bbox_to_anchor=(1, 1), loc=2){% endhighlight %</details>

Kết quả thu được là 1300 mẫu quan sát với tất cả các độ tuổi từ mới sinh đến 65 tuổi. Các biến này tương ứng với các biến trong DAG trên, và mô phỏng này tuân theo DAG.

Tôi đã thể hiện data này trong [**HÌNH 6.4**](#f4), mỗi cá nhân là một điểm. Điểm màu đỏ là các nhân đã kết hôn. Tuổi nằm ở trục hoành, và hạnh phúc ở trục tung, với những người hạnh phúc nhất nằm trên cùng. Ở tuổi 18, họ có thể kết hôn, và dần dần nhiều cá thể sẽ kết hôn mỗi năm. Cho nên ở tuổi lớn hơn, nhiều người kết hôn hơn. Nhưng ở toàn bộ các độ tuổi, người hạnh phúc nhất sẽ dễ kết hôn hơn.

Giả sử bạn gặp data này và muốn hỏi rằng tuổi tác có liên quan đến hạnh phúc không. Bạn không biết mô hình nhân quả thực sự. Nhưng bạn lý lẽ rằng, tình trạng hôn nhân có thể là biến gây nhiễu. Nếu những người kết hôn có hạnh phúc hơn hay kém, trên trung bình, thì bạn cần phải đặt điều kiện trên tình trạng hôn nhân để suy luận quan hệ giữa tuổi và hạnh phúc.

Hãy thử hồi quy đa biến nhằm vào suy luận ảnh hưởng của tuổi tác trên hạnh phúc, khi kiểm soát tình trạng hôn nhân. Đây là một hồi quy đa biến bình thường, giống như những mô hình khác trong chương này và chương trước. Mô hình tuyến tính là đây:

$$ \mu_i =\alpha_{\text{MID}[i]} + \beta_A A_i $$

Trong đó `MID[i]` là chỉ số cho tình trạng hôn nhâ có cá thể $i$, với 0 là đơn thân, 2 là đã kết hôn. Đây chỉ là chiến thuật biến phân nhóm từ Chương 4. Tạo prior dễ hơn, khi chúng ta dùng nhiều intercept, mỗi một cái cho mỗi nhóm, hơn là chúng ta dùng biến chỉ điểm.

Bây giờ chúng ta nên làm nhiệm vụ của mình và nghĩ về các prior. Hãy xem xét slope $\beta_A$ trước, bởi vì cách chúng ta chỉnh thang đo của biến $A$ sẽ quyết định ý nghĩa của intercept. Chúng ta sẽ tập trung chỉ vào những mẫu người trưởng thành, những người lớn hơn 18 tuổi. Tưởng tượng rằng có một mối quan hệ mạnh giữa tuổi tác và hạnh phúc, thì hạnh phúc này là tối đa vào lúc 18 tuổi giảm dần đến thấp nhất lúc 65 tuổi. Sẽ dễ hơn nếu chúng ta chỉnh thang đo tuổi tác để nó có khoảng từ 18 đến 65 là một đơn vị. Code này sẽ thực hiện điều đó:

<b>code 6.22</b>
```python
d2 = d[d.age > 17].copy()  # only adults
d2["A"] = (d2.age - 18) / (65 - 18)
```

Biến $A$ mới sẽ có giá trị từ 0 đến 1, 0 tương ứng với tuổi 18 và 1 là tuổi 65. Hạnh phúc sẽ ở thang đo khác, trong data này, từ -2 đến +2. Cho nên mối quan hệ tưởng tượng mạnh nhất, có hạnh phúc từ tối đa đến tối thiểu, có slope trải dài từ $(2-(-2))/1=4$. Nhớ rẳng 95% mật độ xác suất của phân phối normal là chứa trong vòng 2 đơn vị độ lệch chuẩn. Cho nên nếu chúng ta đặt độ lệch chuẩn của prior thành một nửa của 4, chúng ta đang nói rằng chúng ta mong đợi 95% các slope phù hợp sẽ nhỏ hơn quan hệ mạnh tối đa. Đây không phải là prior mạnh, nhưng lần nữa, nó chí ít giúp giới hạn suy luận trong khoảng thực tế. Bây giờ đến intercept. Mỗi $\alpha$ là giá trị $\mu_i$ khi $A_i =0$. Trong trường hợp này, có nghĩa là tuổi 18. Vậy chúng ta cần phải cho phép $\alpha$ chấp nhận toàn bộ khoảng của điểm hạnh phúc. Normal(0, 1) sẽ đặt 95% mật độ trong khoảng từ -2 đến +2.

Cuối cùng, hãy ước lượng posterior. Chúng ta cần phải xây dựng biến chỉ số tình trạng hôn nhân. Tôi sẽ làm điều đó, và sau đó chạy `SVI`.

<b>code 6.23</b>
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
p6_9, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_9.sample_posterior(random.PRNGKey(1), p6_9, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
 a[0]  -0.20  0.06   -0.20  -0.30  -0.10  1049.96   1.00
 a[1]   1.23  0.09    1.23   1.09   1.37   898.97   1.00
   bA  -0.69  0.11   -0.69  -0.88  -0.53  1126.51   1.00
sigma   1.02  0.02    1.02   0.98   1.05   966.00   1.00</samp>

Mô hình khá khẳng định rằng tuổi tương quan âm với hạnh phúc. Chúng ta muốn so sánh suy luận từ mô hình này với mô hình mà không có tình trạng hôn nhân. Nó đây, theo sau đó là so sánh giữa các phân phối posterior biên:

<b>code 6.24</b>
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
p6_10, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_10.sample_posterior(random.PRNGKey(1), p6_10, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%   n_eff  r_hat
    a   0.01  0.08    0.01  -0.12   0.12  931.50   1.00
   bA  -0.01  0.13   -0.01  -0.22   0.21  940.88   1.00
sigma   1.21  0.03    1.21   1.17   1.26  949.78   1.00</samp>

Mô hình này thì ngược lại, không có quan hệ giữa tuổi tác và hạnh phúc.

Hiện tượng này là chính xác những gì chúng ta sẽ gặp nếu đặt điều kiện trên biến xung đột. Biến xung đột ở đây là tình trạng hôn nhân. Nó là một hệ quả chung của tuổi tác và hạnh phúc. Kết quả là, khi chúng ta đặt điều kiện trên nó, chúng ta tạo ra một mối quan hệ giả tạo giữa hai nguồn căn nguyên. Cho nên nó giống như là, để mô hình hoá `m6_9`, là tuổi tác tương quan âm với hạnh phúc. Nhưng đây chỉ là một quan hệ thống kê, không phải quan hệ nhân quả. Một khi chúng ta biết ai đó là kết hôn hay chưa, biết thêm tuổi tác sẽ cung cấp thêm thông tin về họ hạnh phúc như thế nào.

Bạn có thể thấy được hiện tượng này ở [**HÌNH 6.4**](#f4). Nhìn vào các điểm màu đỏ, những người đã kết hôn. Trong các điểm màu đỏ, người lớn tuổi thì ít hạnh phúc hơn. Đó là bởi vì theo thời gian nhiều người kết hôn hơn, và trung bình của hạnh phúc trong nhóm người kết hôn sẽ tiệm cận với hạnh phúc trung bình quần thể. Nhìn vào các điểm màu xanh, những người chưa kết hôn. Điều đó vẫn đúng khi hạnh phúc giảm dần theo độ tuổi. Đó là bởi vì những người có giá trị hạnh phúc cao hơn đã dần dần đi qua phần điểm màu đỏ. Cho nên cả hai quần thể chưa kết hôn và đã kết hôn, có một mối tương quan âm giữa tuổi tác và hạnh phúc. Nhưng không quần thể nào phản ảnh đúng quan hệ nhân quả.

Trong ví dụ này này thì nó dễ được nhận ra. Kết hôn có nên ảnh hưởng hạnh phúc? Giả sử hạnh phục thực sự thay đổi theo tuổi tác? Nhưng điều đó không liên quan đến vấn đề chính. Nếu bạn không có mô hình nhân quả, bạn không thể tạo suy luận từ hồi quy đa biến. Và hồi quy bản thân nó không cung cấp các bằng chứng cần thiết cho mô hình nhân quả. Thực vậy, bạn cần đến khoa học.

### 6.3.2 DAG bị ám

Sai lệch xung đột xuất phát từ việc đặt điều kiện trên một hệ quả chung, như ví dụ trước. Nếu có thể dựng sơ đồ nhân quả, chúng ta có thể tránh được điều này. Nhưng việc phát hiện một biến xung đột tiềm năng không dễ dàng chút nào, bởi vì còn có nhiều nguồn căn nguyên không đo đạc được. Nguồn căn nguyên không đo đạc được vẫn có thể gây ra sai lệch xung đột. Cho nên tôi xin lỗi và nói rằng chúng ta cũng phải suy nghĩ đến khả năng là DAG của chúng ta đã bị ám.

Giả sử chúng ta muốn suy luận ảnh hưởng của cả cha mẹ ($P$) và ông bà ($G$) lên thành tích giáo dục của con cái ($C$). Bởi vì ông bà được giả định ảnh hưởng đến giáo dục con của họ, có mũi tên từ $G \to P$. Đến đây điều này có vẻ dễ dàng. Nó giống như cấu trúc của ví dụ tỉ suất ly dị từ chương trước:

![](/assets/images/dag 6-6.svg)

Nhưng giả sử có thêm một yếu tố chung không đo đạc được, ảnh hưởng đến cả cha mẹ lẫn con cái, như yếu tố hàng xóm, nhưng lại không ảnh hưởng đến ông bà (những người sống ở bờ nam nước Tây Ban Nha). Lúc đó DAG của chúng ta bị ám bởi biến $U$ không được quan sát:

![](/assets/images/dag 6-7.svg)

Bây giờ $P$ là kết quả chung của $G$ và $U$, cho nên nếu ta đặt điều kiện trên $P$, nó sẽ làm sai lệch suy luận về $G \to C$, *cho dù chúng ta không bao giờ đo đạc được $U$*. Tôi không mong đợi sự thật này sẽ rõ ràng ngay lúc này. Cho nên hãy tiếp tục qua một ví dụ định lượng.

Đầu tiên, hãy mô phỏng thử 200 cặp ba ông bà, cha mẹ, và con cái. Mô phỏng này thì dễ. Chúng ta chỉ cần ánh xạ DAG của chúng ta vào một chuỗi các quan hệ chức năng. DAG trên suy ra rằng:

1. $P$ là một hàm số của $G$ và $U$
2. $C$ là một hàm số của $G$, $P$, và $U$
3. $G$ và $U$ không phải hàm số của những biến số đã biết khác

Chúng ta có thể cho những gợi ý năng thành một mô phỏng đơn giản, sử dụng `dist.Normal` để tạo ra những mẫu quan sát. Nhưng để làm điều này, chúng ta cần phải chính xác hơn là "một hàm số". Cho nên tôi sẽ tạo ra một vài độ mạnh của các quan hệ:

<b>code 6.25</b>
```python
N = 200  # number of grandparent-parent-child triads
b_GP = 1  # direct effect of G on P
b_GC = 0  # direct effect of G on C
b_PC = 1  # direct effect of P on C
b_U = 2  # direct effect of U on P and C
```

Những tham số này giống như slope của mô hình hồi quy. Chú ý rằng tôi giả định là ông bà $G$ có zero hiệu ứng đến con cháu $C$. Ví dụ này không phụ thuộc vào hiệu ứng chính xác bằng zero, nhưng chỉ để bài học rõ ràng hơn. Bây giờ chúng ta dùng những slope này để tạo mẫu ngẫu nhiên:

<b>code 6.26</b>
```python
with numpyro.handlers.seed(rng_seed=1):
    U = 2 * numpyro.sample("U", dist.Bernoulli(0.5).expand([N])) - 1
    G = numpyro.sample("G", dist.Normal().expand([N]))
    P = numpyro.sample("P", dist.Normal(b_GP * G + b_U * U))
    C = numpyro.sample("C", dist.Normal(b_PC * P + b_GC * G + b_U * U))
    d = pd.DataFrame({"C": C, "P": P, "G": G, "U": U})
```

Tôi đã làm cho hiệu ứng cho hàng xóm, $U$, là nhị phân. Điều này sẽ làm cho ví dụ dễ hiểu hơn. Nhưng ví dụ không phụ thuộc vào giả định. Những dòng khác chỉ là mô hình tuyến tính dưới dạng `dist.Normal`.

Bây giờ chuyện gì sẽ xảy ra nếu chúng ta sẽ suy luận ảnh hưởng của ông bà? Bởi vì một vài của toàn bộ hiệu ứng ông bà truyền gián tiếp đến con cái thông qua cha mẹ, chúng ta nhận ra chúng ta cần kiểm soát biến cha mẹ. Đây là một hồi quy đơn giản của $C$ trên $P$ và $G$. Thông thường tôi sẽ khuyến cáo chuẩn hoá các biến, bởi vì nó giúp thành lập các prior hợp lý dễ dàng hơn. Nhưng tôi sẽ giữa data mô phỏng này ở thang đo gốc, để bạn có thể thấy được suy luận như thế nào về các slope trên. Nếu chúng ta thay đổi thang đo, chúng ta sẽ không mong đợi có những giá trị đó lại. Nhưng nếu chúng ta giữ nguyên thang đo, chúng ta có thể sẽ lấy lại thứ gì đó gần với những giá trị này. Cho nên tôi xin lỗi vì dùng prior mơ hồ ở đây, chỉ để tiếp tục ví dụ.

<b>code 6.27</b>
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
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    a  -0.08  0.10   -0.09  -0.24   0.06  1049.96   1.00
 b_GC  -0.71  0.11   -0.71  -0.89  -0.55   813.76   1.00
 b_PC   1.72  0.04    1.72   1.65   1.79   982.64   1.00
sigma   1.39  0.07    1.39   1.28   1.49   968.54   1.00</samp>

Hiệu ứng được suy luận ra của cha mẹ khá cao, lớn gấp 2 lần hơn giá trị mô phỏng của nó. Không có gì ngạc nhiên. Vài tương quan giữa $P$ và $C$ còn do $U$, và mô hình thì không biết $U$. Đây là một hiện tượng nhiễu đơn giản. Ngạc nhiên hơn là mô hình tin rằng hiệu ứng trực tiếp của ông bà là gây tổn hại đến con cháu của họ. Mô hình tuyến tính không sai. Nhưng diễn giải nhân quả của quan hệ này có vấn đề.

<a name="f5"></a>![](/assets/images/fig 6-5.svg)
<details class="fig"><summary>Hình 6.5: Hiện tượng nhiễu không được quan sát và sai lệch xung đột. Trong ví dụ này, ông bà ảnh hưởng con cháu chỉ qua con đường gián tiếp qua cha mẹ. Tuy nhiên, hiệu ứng hàng xóm không quan sát được lên cha mẹ và con cái của họ tạo ra ảo giác rằng ông bà gây hại cho giáo dục con cháu. Giáo dục cha mẹ là một biến xung đột: Một khi chúng ta đặt điều kiện trên nó, giáo dục ông bà trở hành tương quan âm với giáo dục con cháu.</summary>
{% highlight python %}plt.scatter(d[d['U']==-1]['G'], d[d['U']==-1]['C'], edgecolor='C0',s=50, facecolor='white')
plt.scatter(d[d['U']==1]['G'], d[d['U']==1]['C'], edgecolor='C1', s=50, facecolor='white')
g_seq = np.linspace(-3,3,100)
plt.plot(g_seq, g_seq*-0.71)
lb, up = np.quantile(d['P'], q=[0.45, 0.60])
d2 = d[(d['P']>lb) & (d['P']< up)]
plt.scatter(d2[d2['U']==-1]['G'], d2[d2['U']==-1]['C'], edgecolor='C0',s=50)
plt.scatter(d2[d2['U']==1]['G'], d2[d2['U']==1]['C'], edgecolor='C1', s=50)
plt.gca().set(xlabel="giáo dục của ông bà (G)", ylabel="giáo dục của con cái(C)")
plt.annotate('Hàng xóm tốt', (-2,8), color='C1')
plt.annotate('Hàng xóm xấu', (1,-8), color='C0')
plt.text(-2.5, 10, "Cha mẹ ở khoảng 45 đến 60 percentile"){% endhighlight %}</details>

Vậy sai lệch xung đột xuất phát từ đâu trong trường hợp này? Nhìn vào [**HÌNH 6.5**](#f5). Trục hoành là giáo dục của ông bà, trục tung là giáo dục của con cháu. Có 2 đám mây các điểm. Điểm màu đỏ là con cái ở môi trường tốt ($U=1$). Điểm xanh là con cái ở môi trường xấu ($U=-1$). Nhìn tổng thể cả hai đám mây các điểm thì thấy tương quan dương giữa $G$ và $C$. Những người ông bà giáo dục tốt sẽ có những người cháu được giáo dục tốt, nhưng toàn bộ hiệu ứng này là thông qua cha mẹ. Tại sao? Bởi vì data này theo mô phỏng của chúng ta. Hiệu ứng của $G$ trong mô phỏng là zero.

Vậy tương quan âm này từ đâu, khi chúng ta đặt điều kiện lên cha mẹ? Đặt điều kiện lên cha mẹ giống như chọn ra trong nhóm cha mẹ giống nhau về giáo dục. Hãy thử điều đó. Trong [**HÌNH 6.5**]($f5), tôi đã tô màu những cha mệ giữa khoảng giáo dục từ 45 đến 60 percentile. Không có gì đặc biệt về hoảng này. Nó chỉ giúp hiện tượng được dễ nhận ra hơn. Bây giờ nếu chúng ta vẽ đường hồi quy chỉ bằng những điểm này, hồi quy $C$ trên $G$, slope sẽ là số âm. Sẽ có tương quan âm trong hồi quy đa biến này. Tại sao như vậy?

Nó tồn tại bởi vì khi ta biết $P$, biết thêm $G$ vô tình nói cho chúng ta biết về hàng xóm $U$, và $U$ liên quan đến kết cục $C$. Tôi biết điều này rất khó hiểu. Như tôi luôn nói, nếu bạn thấy hoang mang, đó là bởi vì bạn đang tập trung. Hãy xem xét hai cha mẹ khác nhau có cùng mức độ giáo dục, ví dụ ở điểm trung vị 50 percentile. Một trong những cha mẹ này có ông bà có giáo dục tốt. Những cha mẹ khác thì có ông bà ít giáo dục hơn. Cách duy nhất có thể, trong ví dụ này, để cha mẹ có cùng mức giáo dục là họ sống trong môi trường hàng xóm khác nhau. Chúng ta không thể thấy hiệu ứng của hàng xóm - chúng ta chưa đo lường nó, hãy nhớ lại - nhưng ảnh hưởng của hàng xóm vấn truyền trải qua con cháu $C$. Cho nên hai cha mẹ huyền ảo này có cùng mức giáo dụ, người có ông bà giáo dục tố trở nên có người cháu ít giáo dục hơn. Người có ông bà ít giáo dục lại trở thành con người cháu giáo dục tốt hơn. $G$ dự đoán $C$ thấp hơn. 

Biến $U$ không đo lường được làm cho $P$ thành biến xung đột, và đặt điều kiện lên $P$ tạo ra sai lệch xung đột. Vậy chúng ta có thểlàm gì? Bạn phải đo lường $U$. Đây là hồi quy có đặt điều kiện trên $U$:

<b>code 6.28</b>
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
p6_12, losses = svi.run(random.PRNGKey(0), 1000)
post = m6_12.sample_posterior(random.PRNGKey(1), p6_12, (1000,))
print_summary(post, 0.89, False)
```
<samp>        mean   std  median   5.5%  94.5%    n_eff  r_hat
    U   1.87  0.17    1.88   1.59   2.11  1009.20   1.00
    a  -0.06  0.08   -0.05  -0.18   0.07   766.03   1.00
 b_GC   0.01  0.10    0.01  -0.15   0.17  1031.98   1.00
 b_PC   0.99  0.07    0.99   0.88   1.11  1106.62   1.00
sigma   1.08  0.05    1.08   0.99   1.16   797.58   1.00</samp>

Và những slope này phù hợp với data mà chúng ta mô phỏng.

<div class="alert alert-info">
<p><strong>Nghịch lý thống kê và giải thích nhân quả.</strong> Ví dụ ông bà là một ví dụ điển hình của <strong>NGHỊCH LÝ SIMPSON</strong>. Việc thêm biến dự đoán mới ($P$ trong ví dụ này) làm đảo dấu tương quan giữa vài biến dự đoán ($G$) với biến kết cục ($C$). Thông thường, nghịc lý Simpson được trình bày dưới dạng thêm biến mới là có ích. Nhưng trong trường hợp này, nó gây chúng ta hiểu sai. Nghịch lý Simpson là một hiện tượng thống kê. Để biết sự đảo dấu tương quan này có phản ánh chính quan hệ nhân quả, chúng ta cần thứ gì đó hơn là chỉ mô hình thống kê.</p></div>

## <center>6.4 Đối phó với nhiễu</center><a name="a4"></a>

Trong chương này và chương trước, có nhiều ví dụ về cách chúng ta dùng hồi quy đa biến để đối phó với nhiễu. Nhưng chúng ta cũng thấy hồi quy đa biến cũng có thể *gây* ra nhiễu - kiểm soát sai biến sẽ tàn phá suy luận. Hi vọng rằng tôi đã thành công doạ các bạn để biết sợ hãi việc thêm tất cả mọi thứ vào mô hình và hi vọng hồi quy tự xử lý, cũng như khích lệ các bạn tin rằng suy luận hiệu quả là có thể, nếu chúng ta cẩn thận và tự trang bị đủ kiến thức.

Nhưng nguyên tắc nào giải thích cho việc đôi khi thêm hoặc bỏ các biến có thể tạo ra cùng một hiện tượng? Có những quái vật nhân quả nào khác ở ngoài kia, ám ảnh sơ đồ của chúng ta? Chúng ta cần thêm vài nguyên tắc để gộp các ví dụ này lại.

Hãy định nghĩa **NHIỄU (CONFOUNDING)** là trong bất kỳ bối cảnh mà trong đó quan hệ giữa kết cục $Y$ và biến dự đoán quan tâm $X$ không giống như vón dĩ của nó, nếu chúng ta thí nghiệm quyết định các giá trị của $X$. Ví dụ , giả sử chúng ta quan tâm đến quan hệ giữa đào tạo $E$ và bậc lương $W$. Vấn đề là trong một quần thể điển hình có rất nhiều biến không được quan sát $U$ ảnh hưởng cả $E$ và $W$. Ví dụ bao gồm nơi người đó ở, cha mẹ là ai, và bạn bè họ là ai. DAG sẽ trông giống như vậy:

![](/assets/images/dag 6-8.svg)

Nếu ta hồi quy $W$ trên $E$, ước lượng hiệu ứng nhân quả sẽ bị nhiễu bởi $U$. Nó bị nhiễu, bởi vì có 2 con đường nối giữa $E$ và $W$: (1) $E \to W$
và (2) $E \gets U \to W$. Một "con đường" ở đây nghĩa là bất kỳ dãy các biến số mà bạn có thể đi qua từ một biến đến một biến khác, bỏ qua hướng của các mũi tên. Cả hai con đường này nếu tạo tương quan thống kê giữa $E$ và $W$. Nhưng chỉ có con đường đầu tiên là nhân quả, con đường thứ hai là không nhân quả. Tại sao? Bởi vì nếu con đường thứ hai tồn tại, và chúng ta thay đổi $E$, nó sẽ không thay đổi $W$. Toàn bộ hiệu ứng nhân quả của $E$ trên $W$ hoạt động chỉ trên con đường thứ nhất.

Làm sao để cách ly con đường nhân quả? Giải pháp nổi tiếng nhất là chạy nghiên cứu can thiệp. Nếu chúng ta có thể gán mức giáo dục ngẫu nhiên, nó thay đổi DAG:

![](/assets/images/dag 6-9.svg)

Sự kiểm soát loại bỏ ảnh hưởng của $U$ trên $E$. Biến không được quan sát không ảnh hưởng đến giáo dục khi chúng ta tự quyết định giáo dục. Với sự loại bỏ ảnh hưởng từ $U$ trên $E$, con đường $E \gets U \to W$ bị mất đi. Nó chặn con đường thứ hai. Khi con đường bị chặn, chỉ còn một con đường để thông tin đi từ $E$ đến $W$, và sau đó đo lường hiệu ứng giữa $E$ và $W$ sẽ cho một ước lượng tốt cho suy luận nhân quả. Sự kiểm soát loại trừ nhiễu, bởi vì nó chặn những con đường khác từ $E$ sang $W$.

May mắn thay, có phương pháp thống kê học để đạt được điều này, mà không cần thực sự kiểm soát $E$. Cách nào? Cách rõ ràng nhất là thêm $U$ vào mô hình, đặt điều kiện trên $U$. Tại sao điều này lại loại bỏ được nhiễu? Bởi vì nó chặn dòng chảy thông tin giữa $E$ và $W$ thông qua $U$. Nó chặn con đường thứ hai.

Để hiểu tại sao đặt điều kiện trên $U$ chặn con đường $E \gets U \to W$, bạn cần nghĩ con đường này là một mô hình độc lập khác. Khi bạn biết $U$, biết thêm $E$ không cho thông tin gì thêm về $W$. Giả sử $U$ là mức độ giàu có trung bình tại một vùng. Vùng giàu có hơn có nhiều trường tốt hơn, dẫn đến giáo dục $E$ tốt hơn, cũng như công việc có lương $W$ khá hơn. Nếu bạn không biết vùng mà người đó đang sống, biết được giáo dục $E$ của người đó sẽ cho thêm thông tin về mức lương $W$, bởi vì $E$ và $W$ đều tương quan với vùng miền sinh sống. Nhưng sau khi bạn biết được vùng sinh sống của người đó, giả sử không còn đường nào khác giữa $E$ và $W$, thì biết thêm  $E$ sẽ không cho thông tin thêm về $W$. Điều này cũng giống như đặt điều kiện trên $U$ sẽ chặn đường - nó làm cho $E$ và $W$ độc lập, với điều kiện trên $U$.

### 6.4.1. Chặn của sau

Chặn các con đường gây nhiễu giữa vài biến dự đoán $X$ và biến kết cục $Y$ còn gọi là chặn **CỬA SAU (BACKDOOR)**. Chúng ta không muốn có quan hệ giả tạo nào len lỏi trong những con đường không phải nhân quả mà đi vào sau lưng biến dự đoán $X$. Trong ví dụ trên, con đường $E \gets U \to W$ là backdoor, bởi nó vào $E$ bằng mũi tên và kết nối $E$ với $W$. Con đường này là không mang tính nhân quả - can thiệp trên $E$ sẽ không gây thay đổi $W$ qua con đường này - nhưng nó vẫn tạo tương quan giữa $E$ và $W$.

Có một tin tốt là, với một sơ đồ nhân quả DAG, luôn luôn có thể phát hiện, nếu có bất kỳ, các biến nào phải kiểm soát để chặn các con đường backdoor. Nó cũng có thể phát hiện biến nào mà chúng không được kiểm soát, để tránh tạo ra nhiễu mới. Và - tin tốt hơn nữa - chỉ có bốn loại quan hệ giữa các biến để kết hợp lại tạo thành mọi DAG khả dĩ. Cho nên bạn chỉ cần hiểu bốn món này và cách thông tin lan truyền trong chúng. Tôi sẽ định nghĩa bốn loại quan hệ này. Sau đó sẽ thực hành trên ví dụ.

<a name="f1"></a>![](/assets/images/fig 6-6.svg)
<details class="fig"><summary>Hình 6.6: Bốn nguyên tố tạo gây nhiễu. Bất kỳ DAG nào cũng được xây dựng trên những quan hệ cơ bản này. Từ trái sang phải: $X \perp\\!\\!\perp Y \|Z$ trong Fork và Pipe, $X \perp\\!\\!\\!\not{}\\!\\!\\!\perp Y \|Z$ trong Collider, và điều kiện trên Descendant D giống như đặt điều kiện trên cha $Z$.</summary>
{% highlight python %}dag = CausalGraphicalModel(
    nodes=["X","Y",'Z',"X1","Z1","Y1","X2","Z2","Y2","X3","Z3","Y3","D3"],
    edges=[("Z","X"), ("Z","Y"),("X1","Z1"),("Z1","Y1"),("X2","Z2"),("Y2","Z2"),("X3","Z3"),("Z3","D3"),("Y3","Z3")]
)
pgm = daft.PGM()
coordinates = {"X":(0,0),"Z":(1,2),"Y":(2,0),"X1":(3,0),"Z1":(4,1),"Y1":(5,2),"X2":(6,2),"Z2":(7,0),"Y2":(8,2),"X3":(9,2),"Z3":(10,0),"D3":(10,1.5),"Y3":(11,2)}
for node in dag.dag.nodes:
    pgm.add_node(node, node[0], *coordinates[node])
for edge in dag.dag.edges:
    pgm.add_edge(*edge)
pgm.add_text(0.5,-0.5, "The Fork")
pgm.add_text(3.5,-0.5, "The Pipe")
pgm.add_text(6.1,-0.5, "The Collider")
pgm.add_text(8.9,-0.5, "The Descendant")
pgm.render()
plt.gca().invert_yaxis(){% endhighlight %}</details>


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