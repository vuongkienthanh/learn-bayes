---
title: "Chapter 7: Ulysses' Compass"
description: "Chương 7: La bàn của Ulysses"
---

- [7.1 Vấn đề của parameter](#a1)
- [7.2 Entropy và độ chính xác](#a2)
- [7.3 Huấn luyện golem: Regularizing](#a3)
- [7.4 Dự đoán độ chính xác của dự đoán](#a4)
- [7.5 So sánh mô hình](#a5)
- [7.6 Tổng kết](#a6)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import lax, ops, random, vmap
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_value, log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Mikołaj Kopernik (hay Nicolas Copernicus, 1473-1543): Nhà thiên văn học Ba Lan, luật sư Thiên Chúa Giáo, và là kẻ lừa đảo. Nổi tiếng từ mô hình nhật tâm của hệ mặt trời, Kopernik lập luận rằng so với mô hình địa tâm, thì mô hình nhật tâm "hài hoà" hơn. Chính suy nghĩ này dẫn đến (nhiều thập kỷ sau) sự bất đồng và xét xử nổi tiếng bởi Thiên Chúa Giáo của Galileo.

Câu chuyện này là một huyền thoại về khoa học chiến thắng thuyết lý tưởng và mê tín. Nhưng lập luận của Kopernik nhìn vẫn kém, nếu không nói đến sự lý tưởng hoá. Có hai vấn đề: mô hình đó chưa chắc gì hài hoà hay chính xác hơn mô hình địa tâm. Mô hình Copernicus rất phức tạp. Thực vậy, nó cũng có những epicycle rườm rà tương tự như mô hình Ptolemaic ([**HÌNH 7.1**](#f1)). Kopernik đã đưa mặt trời vào trung tâm, nhưng ông ta vẫn dùng những vòng tròn đều để làm quỹ đạo, ông vẫn cần epicycle. Cho nên "sự hài hoà" không thích hợp để mô tả hình dạng của mô hình. Cũng giống như mô hình Ptolemaic, mô hình Kopernicus thực ra cũng là một Fourier series, một phương pháp ước lượng hàm chu kỳ. Điều này dẫn đến vấn đề thứ hai: Mô hình nhật tâm cho dự đoán tương tự như mô hình địa tâm. Ước lượng giống nhau vẫn có thể xảy ra cho dù Trái Đất đứng yên hay chuyển động. Cho nên không có lý do gì để ưu ái một mô hình hơn chỉ dựa vào độ chính xác.

<a name="f1"></a>![](/assets/images/fig 7-1.png)
<details class="fig"><summary>Hình 7.1: Mô hình Ptolemaic (trái) và Copernican (phải) cho hệ mặt trời. Cả mô hình dùng epicycle (vòng tròn trên vòng tròn), và cả hai mô hình đều cho dự đoán giống nhau. Tuy nhiên, mô hình Copernican cần ít vòng tròn hơn. (Không phải tất cả vòng tròn epicycle đều hiện ở hình này.)</summary></details>

Nhưng Kopernik không thích chỉ dựa vào "sự hài hoà" mơ hồ. Ông cũng lập luận cho sự vượt trội của mô hình của ông dựa trên cơ sở cần ít nhân tố hơn: "Do đó, chúng tôi theo Tự nhiên, người không tạo ra thứ gì vô ích hoặc thừa, thường thích tạo ra một nguyên nhân với nhiều hiệu ứng."<sup><a name="r98" href="#98">98</a></sup> Và đúng là mô hình nhật tâm cần ít epicycle hơn để cho dự đoán giống với mô hình địa tâm. Một cách nói khác, nó *đơn giản hơn*.

Nhà khoa học thường thích những giả thuyết đơn giản. Sở thích này đôi khi mơ hồ - một dạng sở thích về thẩm mỹ. Đôi khi chúng ta cần sự thực dụng, việc thích mô hình đơn giản là do mô hình đơn giản của chúng thì dễ sử dụng hơn. Thông thường, nhà khoa học thường trích dẫn một nguyên tắc hời hợt là **LƯỠI DAO CỦA OCKHAM (OCKHAM'S RAZOR)**: *Mô hình với ít giả định hơn thì được ưa chuộng hơn.* Trong trường hợp Kopernik và Ptolemy, nguyên tắc này đưa ra phân minh rõ ràng. Nó không khẳng định Kopernik là đúng (ông ta rõ ràng không đúng), nhưng bởi vì mô hình nhật tâm và địa tâm cho cùng một dự đoán, chí ít lưỡi dao này có thể giải quyết rõ ràng cho tranh luận này. Nhưng vẫn rất khó để sử dụng lưỡi dao này một cách tổng quát, bởi vì thông thường chúng ta thường phải phân biệt mô hình trên sự khác nhau về độ chính xác và sự đơn giản của nó. Làm thế nào để đánh đổi những tiêu chí khác nhau giữa các mô hình này? Lưỡi dao của Ockham không cho hướng dẫn.

Chương này mô tả vài dụng cụ phổ biến nhất để đương đầu với sự đánh đổi này. Vài yếu tố đơn giản thường xuất hiện trong tất cả những dụng cụ này, nên chúng thường được so sánh với lưỡi dao của Ockham. Nhưng mọi công cụ là như nhau trong việc tăng độ chính xác của dự đoán. Chúng khác lưỡi dao ở chỗ, chúng cụ thể hoá việc đánh đổi giữa độ chính xác và sự đơn giản.

Vậy thay vì dùng lưỡi dao của Ockham, hay nghĩ đến La bàn của Ulysess. Ulysses là người anh hùng trong sử thi *Odyssey* của Homer. Trong cuộc du hành của anh, Ulysses phải định hướng con đường thẳng chật hẹp giữa quái thú nhiều đầu Scylla - tấn công từ các núi đá và nuốt chửng các thuỷ thủ - và thuỷ quái Charybdis - kéo thuyền và người vào nghĩa địa dưới nước. Quá gần một trong hai quái vật đều rất nguy hiểm. Trong bối cảnh khoa học, bạn có thể tưởng tượng hai quái vật này đại diện cho hai loại sai số thống kê cơ bản:

1. Quái thú nhiều đầu **OVERFITTING** dẫn đến dự đoán kém chính xác bởi học quá nhiều từ data.
2. Vòng xoáy **UNDERFITTING** dẫn đến dự đoán kém chính xác bởi học quá ít từ data.

Có một con quái vật thứ ba, mà bạn đã gặp ở chương trước - nhiễu. Trong chương này, bạn sẽ gặp những mô hình bị nhiễu có thể thực ra cho dự đoán tốt hơn mô hình đo lường chính xác quan hệ nhân quả. Hệ quả của nó là, khi thiết kế bất kỳ mô hình thống kê, chúng ta phải quyết định muốn tìm hiểu nhân quả hay là dự đoán. Chúng không phải là chung một mục đích, và mỗi mục đích đều có những mô hình khác nhau. Tuy nhiên, để đo lường chính xác hiệu ứng nhân quả, chúng ta vẫn phải chống lại overfitting. Quái vật overfitting và underfitting luôn xuất hiện ở mọi nơi, cho dù mục đích nào.

Nhiệm vụ của chúng ta là định hướng cẩn thận giữa những con quái vật này. Thông thường có hai hướng tiếp cận. Hướng thứ nhất là dùng **REGULARIZING PRIOR** để giữ chân mô hình đừng quá phấn khởi bởi data. Trong phương pháp non-Bayes, chúng được gọi là "penalized likelihood". Hướng tiếp cận thứ hai là dùng thiết bị tính điểm, như **INFORMATION CRITERIA** hoặc **CROSS-VALIDATION**, để mô hình hoá công việc dự đoán và ước lượng độ chính xác của dự đoán. Cả hai hướng tiếp cận đều thường được sử dụng trong khoa học tự nhiên và xã hội. Hơn nữa, chúng có thể - và nên - dùng song song. Hiểu biết cả hai hướng tiếp cận đều đáng đồng tiền, bởi vì bạn sẽ cần chúng tại một thời điểm nào đó.

Trước khi giới thiệu tiêu chuẩn thông tin (information criteria), chương này cũng phải giới thiệu **THUYẾT THÔNG TIN (INFORMATION THEORY)**. Nếu đây là lần đầu tiên bạn gặp thuyết thông tin, nó sẽ có thể rất lạ. Nhưng vài kiến thức về chúng là cần thiết. Khi bạn bắt đầu dùng những tiêu chuẩn thông tin - chương này mô tả AIC, BIC, WAIC, PSIS - bạn sẽ thấy ứng dụng chúng dễ hơn hiểu chúng nhiều. Nó là một lời nguyền, cho nên chương này nhắm đến đấu tranh lại lời nguyền đó, tập trung vào các cơ sở khái niệm, và ứng dụng theo sau.

Cần biết rằng, trước khi bắt đầu, tài liệu này rất khó. Nếu bạn cảm thấy hoang mang, đó là bình thường. Mọi sự hoang mang bạn cảm thấy chứng tỏ não bạn đang cố gắng đón nhận kiến thức. Càng về sau, sự hoang mang sẽ được thay thế bằng tri thức về các hoạt động của overfitting, regularization, và tiêu chuẩn thông tin trong bối cảnh quen thuộc.

<div class="alert alert-info">
<p><strong>Đếm sao.</strong> Cách làm thông thường để chọn mô hình là chọn mô hình nào có tất cả mọi hệ số đều có ý nghĩa thống kê. Nhà thống kê thường gọi đây là <strong>ĐẾM SAO (STARGAZING)</strong>, theo đúng nghĩa đen là tìm quét dấu sao (**) theo sau các con số ước lượng. Một đồng nghiệp của tôi gọi đây là "Chuyến du hành vào không gian (Space Odyssey)", trong sự vinh danh của A.C. Clarke và tác phẩm của ông. Theo suy nghĩ này, mô hình rất nhiều sao là tốt nhất.</p>
<p>Nhưng mô hình đó không phải tốt nhất. Cho dù bạn nghĩ kiểm định thống kê là gì, sử dụng nó để lựa chọn các cấu trúc mô hình khác nhau là một sai lầm, ví dụ như *p*-values không phải được sinh ra để định hướng giữa overfitting và underfitting. Như bạn sẽ thấy trong chương này, những biến dự đoán giúp cải thiện độ chính xác dự đoán không phải lúc nào cũng có ý nghĩa thống kê. Hoặc vẫn có thể xảy ra trường hợp biến dự đoán có ý nghĩa thống kê nhưng không giúp ích gì cho dự đoán. Con số 5% chỉ mang ý nghĩa tiện lợi, chúng ta không nên dựa dẫm trên nó để tối ưu hoá mọi thứ.</p>
</div>

<div class="alert alert-info">
<p><strong>AIC có phải Bayes không?</strong> AIC thường được cho rằng không phải Bayes. Có lý do lịch sử và thống kê cho nhận định này. Về mặt lịch sử, AIC được tạo ra mà không liên quan đến xác suất. Về mặt thống kê, AIC sử dụng ước lượng MAP thay vì toàn bộ posterior, và nó cần prior phẳng. Cho nên nó không giống Bayes. Để củng cố thêm ấn tượng này là sự tồn tại của một đại lượng khác, <strong>TIÊU CHUẨN THÔNG TIN BAYES (BAYESIAN INFORMATION CRITERION - BIC)</strong>. Tuy nhiên, BIC cũng cần prior phẳng và ước lượng MAP, mặc dù nó thực chất không phải "tiêu chuẩn thông tin".</p>
<p>Suy cho cùng, AIC có cách diễn giải rõ ràng và hợp lý dưới xác suất Bayes, và Akaike cùng nhiều người khác đã có cuộc tranh luận kéo dài cho chứng minh thay thế bằng Bayes của quy trình.<sup><a name="r99" href="#99">99</a></sup> Và bạn sẽ thấy rằng, tiêu chuẩn thông tin kiểu Bayes như WAIC sẽ cho kết quả như AIC, nếu đạt được giả định của AIC. Dưới ánh nhìn mới này, ta có thể xem AIC như là một trường hợp đặc biệt của tiêu chuẩn Bayes như WAIC, ngay cả nếu nó không phải được suy ra giống như ban đầu của AIC. Tất cả nhưng ví dụ này là đặc trưng cơ bản của quy trình thống kê: Một quy trình có thể được cùng được quy ra và xác thực từ nhiều góc nhìn, cho dù đôi khi triết lý ban đầu khác nhau.</p>
</div>

## <center>7.1 Vấn đề của parameter</center><a name="a1"></a>

Ở chương trước, ta đã thấy việc thêm biến số (variable) và tham số (parameter) vào mô hình có thể giúp bộc lộ hiệu ứng bị ẩn và cải thiện ước lượng. Bạn cũng đã thấy việc thêm biến số gây tổn thương, trong tình huống ta chưa có mô hình nhân quả đáng tin cậy. Sai lệch đồng căn (collider) là có thật. Nhưng đôi khi chúng ta không quan tâm đến suy luận nhân quả. Có thể chúng ta chỉ mong muốn cho dự đoán tốt. Xem xét ví dụ ông bà - cha mẹ - con cái ở chương trước. Cho toàn bộ các biến vào mô hình sẽ cho một mô hình dự đoán tốt. Sự thật chuyện gì đang xảy ra là không liên quan và không cần hiểu. Vậy nên cho toàn bộ vào mô hình là an toàn?

Câu trả lời là "không". Có hai vấn đề liên quan đến việc thêm biến số. Đầu tiên là chính việc thêm biến số - làm cho mô hình phức tạp hơn - luôn luôn cải thiện sự vừa vặn của data vào mô hình.<sup><a name="r100" href="#100">100</a></sup> "Sự vừa vặn (fit)" ở đây nghĩa là trị số đo lường về sự chính xác của dự đoán ngược vào data dùng để fit mô hình (trị số Goodness-of-Fit). Có nhiều trị số như thế, và chúng đều có nhược điểm riêng. Trong bối cảnh mô hình tuyến tính, $R^2$ là trị số phổ biến nhất. $R^2$ thường được mô tả như "phương sai giải thích được", được định nghĩa là:

$$ R^2 = \frac{\text{var(outcome)} - \text{var(residuals)} }{\text{var(outcome)}} = 1 - \frac{\text{var(residuals)}}{\text{var(outcome)}} $$

$R^2$ được tính toán khá dễ, và rất phổ biến. Giống như những trị số Goodness-of-Fit khác, $R^2$ tăng lên khi càng nhiều biến số được thêm vào. Điều này đúng ngay cả biến số bạn thêm vào là hoàn toàn ngẫu nhiên, không liên quan đến kết cục (outcome). Vì vậy ta không nên chọn giữa các mô hình thông qua đơn thuần các trị số đó.

Thứ hai, trong khi những mô hình phức tạp fit data tốt hơn, chúng thường dự đoán data mới tệ hơn. Mô hình có nhiều parameter có xu hướng *overfit* hơn mô hình đơn giản. Có nghĩa là mô hình phức tạp sẽ rất nhạy cảm với chính data dùng để fit nó, dẫn đến sẽ có tiềm năng lỗi rất lớn khi data tương lai không giống chính xác như data cũ. Nhưng mô hình đơn giản, quá ít parameter, có xu hướng *underfit*, làm cho dự đoán hơn quá hoặc thiếu hụt một cách có hệ thống, cho dù data tương lai có giống hệt data cũ. Cho nên chúng ta không thể ưu ái bên mô hình đơn giản hoặc bên mô hình phức tạp.

Hãy xem xét cả hai vấn đề này trong bối cảnh đơn giản sau.

### 7.1.1 Nhiều parameter luôn cải thiện fit.

**OVERFITTING** xảy ra khi mô hình học quá nhiều từ mẫu quan sát. Có nghĩa là có cả đặc trưng *thường (regular)* và *không thường (irregular)* ở mọi mẫu quan sát. Đặc trưng thường là mục tiêu để học của chúng ta, bởi vì chúng tổng quát hoá tốt và trả lời câu hỏi nghiên cứu. Đặc trưng thường thì hữu ích, dưới lựa chọn có chủ đích của chúng ta. Đặc trưng không thường là bộ phận của data mà không được tổng quát hoá và cho nên gây ta hiểu nhầm.

Overfitting là hiện tượng tự nhiên xảy ra, thật không may. Trong tất cả các mô hình data đã thấy trong sách này, thêm parameter mới vào luôn cải thiện fit của mô hình cho data. Trong Chương 13, bạn sẽ gặp mô hình mà thêm parameter vào sẽ không nhất thiết cải thiện fit cho mẫu, nhưng sẽ cải thiện độ chính xác dự đoán.

Sau đây là một data ví dụ cho overfitting. Data gồm thể tích trung bình của não (brain volume) và cân nặng (body mass) cho 7 loài vượn.<sup><a name="r101" href="#101">101</a></sup>

```python
sppnames = [
    "afarensis",
    "africanus",
    "habilis",
    "boisei",
    "rudolfensis",
    "ergaster",
    "sapiens",
]
brainvolcc = jnp.array([438, 452, 612, 521, 752, 871, 1350])
masskg = jnp.array([37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5])
d = pd.DataFrame({"species": sppnames, "brain": brainvolcc, "mass": masskg})
```

![](/assets/images/fig 7-2.svg)
<details class="fig"><summary>Hình 7.2: Thể tích não trung bình ở đơn vị centimet khối, ứng với trọng lượng cơ thể ở đơn vị kilogram, cho 6 loài vượn. Mô hình nào là tốt nhất để mô thể mối liên hệ giữa kích thước não và cơ thể?</summary>
<pre><code>plt.scatter(brainvolcc, masskg);
for s, x, y in zip(sppnames, brainvolcc, masskg):
    plt.annotate(s, (x, y));</code></pre></details>

Giờ bạn có một dataframe `d` chứa các giá trị của kích thước não (brain volume) và trọng lượng cơ thể (body mass). Data như thế này có độ tương quan rất cao - kích thước não tương quan với kích thước cơ thể xuyên suốt các loài. Một câu hỏi cần thiết, tuy nhiên, là mức độ một loài cụ thể có bộ não lớn hơn so với mong đợi, sau khi xem xét kích thước cơ thể. Một giải pháp thông dụng là fit mô hình hồi quy tuyến tính, nó mô hình hoá kích thước não qua hàm tuyến tính của kích thước cơ thể. Sau đó độ biến thiên của kích thuớc não có thể được mô hình hoá qua hàm của biến khác, như hệ sinh thái và chế độ ăn. Cách tiếp cận này hoàn toàn giống với "kiểm soát thống kê" như chương trước.

Kiểm soát kích thước cơ thể thì, tuy nhiên, phụ thuộc một hàm số rõ ràng cho mối liên hệ giữa kích thước cơ thể và kích thước bộ não. Tới nay chúng ta chỉ dùng hàm tuyến tính. Nhưng tại sao lại dùng một đường thẳng để liên hệ giữa kích thước cơ thể và bộ não? Không có lý do gì mà tự nhiên quy định liên hệ đó của các loài là đường thẳng. Tại sao không nghĩ đến mô hình cong, như parabola? Thực vậy, tại sao là không phải hàm bậc 3, hay spline? Không có lý do gì để giả định tiền nghiệm (prior) là kích thước não bộ tăng dần theo hàm tuyến tính với kích thước cơ thể. Thực vậy, nhiều người đọc thích mô hình tuyến tính giữa log của kích thước não bộ và log của trọng lượng (quan hệ luỹ thừa). Nhưng đây không phải nội dung tôi muốn trình bày trong chương này. Bài học hôm nay vẫn sẽ xuất hiện, cho dù bạn có chuyển đôi data như thế nào.

Hãy fit hàng loạt mô hình có độ phức tạp tăng dần đều, và xem mô hình nào là tốt nhất. Ta sẽ dùng hồi quy đa bậc (polynomial regression). Điều quan trọng cần nhớ là, hồi quy da bậc rất thông dụng, nhưng thường là ý tưởng tối. Trong ví dụ này, tôi sẽ cho bạn thấy đó là ý tưởng tồi nếu dùng nó một cách mù quáng. Spline ở Chương 4 cũng có vấn đề cơ bản tương tự.

Mô hình đơn giản nhất để liên hệ giữa kích thước não và cơ thể là mô hình tuyến tính. Đây sẽ là mô hình đầu tiên ta xem xét. Trước khi viết mô hình, ta sẽ biến đổi (scale, transform) các biến số. Nhớ lại trong những chương trước thì các biến dự đoán và kết cục được scale lại thì giúp mô hình fit dễ hơn và giúp xác định và thấu hiểu prior hơn. Trong trường hợp này, chúng ta muốn chuẩn hoá trọng lượng cơ thể - cho chúng có trung bình (mean) là zero và độ lệch chuẩn (standard deviation std) là 1 - và biến đổi biến kết cục và thể tích não bộ, để chúng có giá trị quan sát lớn nhất là 1. Tại sao là không chuẩn hoá biến thể tích não? Bởi vì chúng ta muốn bảo tồn zero làm giá trị tham khảo: Không có tí não nào. Bạn không thể có não âm. Tôi không nghĩ thế.

```python
d["mass_std"] = (d.mass - d.mass.mean()) / d.mass.std()
d["brain_std"] = d.brain / d.brain.max()
```

Bây giờ đây là phiên bản toán học của mô hình tuyến tính đầu tiên. Mánh khoé cần chú ý
ở đây là prior log-normal của $\sigma$. Nó giúp giữ $\sigma$ luôn luôn dương, theo như cần thiết.

$$\begin{aligned}
b_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta m_i\\
\alpha &\sim \text{Normal}(0.5, 1)\\
\beta &\sim \text{Normal}(0,10)\\
\sigma &\sim \text{Log-Normal}(0,1)\\
\end{aligned}$$

Điều này đơn giản là thể tích trung bình của não bộ $b_i$ của loài $i$ là hàm tuyến tính của trọng lượng cơ thể $m_i$. Bây giờ xét đến những gì prior suy ra. Prior của $\alpha$ tập trung vào trung bình của thể tích não (đã được scale) trong data. Cho nên nó nói rằng loài trung bình với trọng lượng cơ thể trung bình có kích thước não có khoảng tin cậy 89% rơi vào khoảng -1 và 2. Nó khá rộng một cách vô lý và bao gồm cả giá trị âm (không thể xảy ra). Prior của $\beta$ thì phẳng và tập trung ở zero. Nó cho phép mối liên hệ rất lớn cả số dương và số âm. Những prior này cho phép nhưng suy luận phi lý, đặc biệt khi mô hình càng phức tạp hơn. Và đây là phần bài học, ta tiếp tục bằng việc fit mô hình:

```python
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic("mu", a + b * mass_std)
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
m7_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m7_1,
    optim.Adam(0.3),
    Trace_ELBO()
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p7_1 = svi.get_params(state)
```

Tôi dùng `exp(log_sigma)` trong likelihood, và để kết quả luôn luôn lớn hơn zero.

<div class="alert alert-info">
    <p><strong>OLS và hồi quy trong Bayes. </strong>Ta vẫn có thể dùng <strong>BÌNH PHƯƠNG NHỎ NHẤT THÔNG THƯỜNG (ORDINARY LEAST SQUARES - OLS)</strong> để có được posterior cho mô hình kích thước não. Ví dụ, bạn có thể library <code>statsmodel</code> để thực hiện OLS, nhưng bạn không thể có được phân phối posterior của $\sigma$. Các bạn có thể đọc thêm tại đây: <a href="https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html">www.statsmodels.org/stable/examples/notebooks/generated/ols.html</a></p>
    <p>OLS không được xem là thuật toán Bayes. Nhưng chỉ cần prior là mơ hồ, tối thiểu hoá tổng của bình phương độ lệch chuẩn thành đường hồi quy, tương đương với tìm trung bình của posterior. Thật vậy, Carl Friedrich Gauss ban đầu suy ra quy trình OLS từ khung quy trình Bayes.<sup><a name="r102" href="#102">102</a></sup> Trước đó, gần như mọi xác suất đều là Bayes, mặc dù từ "Bayes" không được dùng nhiều cho đến khi thế kỷ 20. Trong nhiều trường hợp, quy trình non-Bayes cũng sẽ có cách diễn đạt tương đương trong Bayes. Sự thật này có ý nghĩa ở hai hướng. Cách diễn đạt Bayes của quy trình non-Bayes nêu ra những giả định ban đầu của thông tin, và điều này rất có ích để hiểu cách nó hoạt động. Ngược lại, mô hình Bayes có thể được thể hiện một cách tương đối bởi quy trình "non-Bayes" với hiệu năng tốt hơn. Suy luận Bayes tức là tìm ra posterior một cách tương đối. Nó không nêu cụ thể cách ra tìm như thế nào.</p>
</div>

Trước khi dừng để vẽ posterior, như chúng ta đã làm ở các chương trước, hãy tập trung vào $R^2$, tỉ lệ variance "giải thích được" bởi mô hình. Nó có nghĩa là mô hình tuyến tính dự đoán ngược vào một bộ phận của toàn bộ sự biến thiên trong kết cục của data mà được fit vào mô hình. Độ biến thiên còn lại chính là phương sai của residual (tồn dư).

Điểm chính của ví dụ này không phải để khen ngợi hoặc chôn vùi $R^2$. Nhưng chúng ta sẽ phải tính nó trước khi chôn nó. Điều này cũng khá dễ. Chúng ta chỉ cần tính phân phối dự đoán posterior (posterior predictive distribution) vào ngược data ban đầu. Sau đó ta trừ mỗi quan sát vào mỗi dự đoán để lấy residual. Sau đó ta cần variance của cả residual và biến kết cục. Nghĩa là ta cần variance thực sự, không phải variance được phần mềm trả về bằng hàm `var`, hàm mà trị số ước lượng theo frequentist và có mẫu số không đúng. Do đó, ta sẽ tính variance bằng thủ công: lấy trung bình của bình phường độ lệch chuẩn từ trung bình. Về nguyên tắc, tiếp cận Bayes cho phép chúng ta làm điều này với mỗi mẫu từ posterior. Nhưng theo truyền thống $R^2$ dùng giá trị trung bình của dự đoán. Cho nên chúng ta ở đây sẽ làm tương tự. Phần sau của chương ta sẽ dùng một trị số Bayes, nó dùng toàn bộ phân phối để tính giá trị của nó.

```python
post = m7_1.sample_posterior(random.PRNGKey(12), p7_1, (1000,))
s = Predictive(m7_1.model, post)(random.PRNGKey(2), d.mass_std.values)
r = jnp.mean(s["brain_std"], 0) - d.brain_std.values
resid_var = jnp.var(r, ddof=1)
outcome_var = jnp.var(d.brain_std.values, ddof=1)
1 - resid_var / outcome_var
```
<samp>0.4774589</samp>

Chúng ta sẽ phải làm điều này cho nhiều mô hình theo sau nữa, do đó ta viết hàm số để làm cho việc này lặp lại dễ hơn.

```python
def R2_is_bad(quap_fit):
    quap, params = quap_fit
    post = quap.sample_posterior(random.PRNGKey(1), params, (1000,))
    s = Predictive(quap.model, post)(random.PRNGKey(2), d.mass_std.values)
    r = jnp.mean(s["brain_std"], 0) - d.brain_std.values
    return 1 - jnp.var(r, ddof=1) / jnp.var(d.brain_std.values, ddof=1)
```

Giờ đến lượt những mô hình để so sánh với `m7_1`. Chúng ta xem xét thêm 5 mô hình, mỗi mô hình sẽ càng phức tạp hơn mô hình trước. Mỗi mô hình sẽ là polynomial và có bậc luỹ thừa cao hơn. Ví dụ, mô hình polynomial bậc 2 liên hệ kích thước cơ thể và não là một parabola. Ở dạng toán học:

$$\begin{aligned}
b_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta_1 m_i + \beta_2 m_i^2 \\
\alpha &\sim \text{Normal} (0.5,1)\\
\beta_j &\sim \text{Normal}(0,10) \quad \text{for } \, j=1..2\\
\sigma &\sim \text{Log-Normal}(0,1)\\
\end{aligned}$$

Họ mô hình này thêm một parameter, $\beta_2$, nhưng sử dụng toàn bộ data như `m7_1`. Để chạy mô hình này, chúng ta định nghĩa $\beta$ là 1 vector.

```python
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([2]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic("mu", a + b[0] * mass_std + b[1] * mass_std ** 2)
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)


m7_2 = AutoLaplaceApproximation(
    model, init_loc_fn=init_to_value(values={"b": jnp.repeat(0.0, 2)})
)
svi = SVI(
    model,
    m7_2,
    optim.Adam(0.3),
    Trace_ELBO(),
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(2000))
p7_2 = svi.get_params(state)
```

Bốn mô hình tiếp theo cũng được xây theo mô típ tương tự. Mô hình `m7_3` đến `m7_6` là bậc 3, bậc 4, bậc 5, bậc 6.

```python
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([3]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic(
        "mu", a + b[0] * mass_std + b[1] * mass_std ** 2 + b[2] * mass_std ** 3
    )
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
m7_3 = AutoLaplaceApproximation(
    model, init_loc_fn=init_to_value(values={"b": jnp.repeat(0.0, 3)})
)
svi = SVI(
    model,
    m7_3,
    optim.Adam(0.01),
    Trace_ELBO(),
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(2000))
p7_3 = svi.get_params(state)
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([4]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic(
        "mu", a + jnp.sum(b * jnp.power(mass_std.reshape(-1, 1), jnp.arange(1, 5)), 1)
    )
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
m7_4 = AutoLaplaceApproximation(
    model, init_loc_fn=init_to_value(values={"b": jnp.repeat(0.0, 4)})
)
svi = SVI(
    model,
    m7_4,
    optim.Adam(0.01),
    Trace_ELBO(),
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(2000))
p7_4 = svi.get_params(state)
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([5]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic(
        "mu", a + jnp.sum(b * jnp.power(mass_std[..., None], jnp.arange(1, 6)), -1)
    )
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
m7_5 = AutoLaplaceApproximation(
    model, init_loc_fn=init_to_value(values={"b": jnp.repeat(0.0, 5)})
)
svi = SVI(
    model,
    m7_5,
    optim.Adam(0.01),
    Trace_ELBO(),
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(2000))
p7_5 = svi.get_params(state)
```

Mô hình `m7_6`, cần thêm một bước đệm. Độ lệch chuẩn được thay thế bằng hằng số 0.001. Mô hình sẽ không hoạt động được nếu không làm bước này, bởi một lý do vô cùng quan trọng mà chúng ta sẽ thấy rõ nếu chúng ta vẽ hình lên. Đây là mô hình cuối:

```python
def model(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([6]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic(
        "mu", a + jnp.sum(b * jnp.power(mass_std[..., None], jnp.arange(1, 7)), -1)
    )
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
m7_6 = AutoLaplaceApproximation(
    model, init_loc_fn=init_to_value(values={"b": jnp.repeat(0.0, 6)})
)
svi = SVI(
    model,
    m7_6,
    optim.Adam(0.003),
    Trace_ELBO(),
    mass_std=d.mass_std.values,
    brain_std=d.brain_std.values,
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(5000))
p7_6 = svi.get_params(state)
```

Giờ ta sẽ vẽ từng mô hình. Chúng ta sẽ là giống như những chương trước: trích mẫu từ posterior, tính ra phân phối posterior dự đoán tại mỗi vị trí trên trục hoành, tổng kết lại và thể hiện biểu đồ. Với mô hình `m7_1`:

```python
post = m7_1.sample_posterior(random.PRNGKey(1), p7_1, (1000,))
mass_seq = jnp.linspace(d.mass_std.min(), d.mass_std.max(), num=100)
l = Predictive(m7_1.model, post, return_sites=["mu"])(
    random.PRNGKey(2), mass_std=mass_seq
)["mu"]
mu = jnp.mean(l, 0)
ci = jnp.percentile(l, (4.5, 95.5), 0)
az.plot_pair(d[["mass_std", "brain_std"]].to_dict("list"))
plt.plot(mass_seq, mu, "k")
plt.fill_between(mass_seq, ci[0], ci[1], color="k", alpha=0.2)
plt.title("m7_1: R^2 = {:0.2f}".format(R2_is_bad((m7_1, p7_1)).item()))
plt.show()
```

Tôi vẽ biểu đồ này cùng với những mô hình khác, với một vài yếu tố thẩm mỹ. Mỗi hình có cho thêm $R^2$. Khi bậc luỹ thừa của trung bình tăng lên, $R^2$ luôn tăng lên, cho thấy rằng có sự cải thiện của dự đoán ngược của data. Mô hình bậc 5 có giá trị $R^2$ là 0.99. Nó hầu như đi qua tất cả mọi điểm. Mô hình bậc 6 thì đi qua toàn bộ điểm data, và residual không có variance. Nó là mức "vừa vặn" hoàn hảo, $R^2=1$. Đó là lý do tại sao chúng ta phải giữ giá trị `sigma` - Nếu nó được ước lượng, nó sẽ thu lại thành zero, bởi vì variance của residual là zero khi đường thẳng đi ngay qua trung tâm của mỗi điểm.

![](/assets/images/fig 7-3.svg)
<details class="fig"><summary>Hình 7.3: Những mô hình tuyến tính polynomial với bậc tăng dần của data các loài vượn. Mỗi hình cho trung bình posterior bằng màu đen, KTC 89% của trung bình màu xám. $R^2$ cũng được cho vào hình.</summary>
<pre><code>fig, ax = plt.subplots(3,2, figsize=(11,18))
m_list = [m7_1, m7_2, m7_3, m7_4, m7_5, m7_6]
p_list = [p7_1, p7_2, p7_3, p7_4, p7_5, p7_6]
for i,(m,p,a) in enumerate(zip(m_list, p_list, ax.flatten())):
    post = m.sample_posterior(random.PRNGKey(1), p, (1000,))
    mass_seq = jnp.linspace(d.mass_std.min(), d.mass_std.max(), num=100)
    l = Predictive(m.model, post, return_sites=["mu"])(
        random.PRNGKey(2), mass_std=mass_seq
    )["mu"]
    mu = jnp.mean(l, 0)
    ci = jnp.percentile(l, (4.5, 95.5), 0)
    az.plot_pair(d[["mass", "brain"]].to_dict("list"), ax=a, scatter_kwargs={"markersize":10, "marker":"o"})
    a.plot(mass_seq*d.mass.std() +d.mass.mean(), mu*d.brain.max(), "k")
    a.fill_between(mass_seq*d.mass.std() +d.mass.mean(), ci[0]*d.brain.max(), ci[1]*d.brain.max(), color="k", alpha=0.2)
    a.set_title("m7_{}: R^2 = {:0.2f}".format(i,R2_is_bad((m, p)).item()));
    if i <5:
        a.set_ylim(0)
    else:
        a.hlines(0,d.mass.min(), d.mass.max(), linestyles='dashed', linewidth=1.2)
    a.set_xlabel('body mass(kg)')
    a.set_ylabel('brain volume(cc)')
plt.tight_layout()</code></pre></details>

Tuy nhiên, bạn có thể thấy từ hình rằng, con đường trung bình các điểm dự đoán ở mô hình bậc càng cao thì càng phi lý. Sự phi lý dễ thấy nhất ở `m7_6`, mô hình phức tạp nhất. Việc fit thì rất hoàn hảo, nhưng mô hình rất lố bịch. Chú ý rằng có khoảng hở giữa data trọng luọng cơ thể, bởi vì không có loài vượn nào có trọng lượng giữa 55 Kg đến khoảng 60 Kg. Ở vùng này, thể tích não trung bình của dự đoán ở mô hình polynomial bậc cao không có gì để dự đoán, và cho nên mô hình không quan tâm đến việc đu đưa dã man ở khoảng này. Biên độ đu đưa lớn đến mức tôi phải mở rộng trục tung để thể hiện độ sâu mà trung bình dự đoán quay đầu lại. Ở khoảng 58 kg, mô hình dự đoán kích thước não là số âm! Mô hình (vẫn) không quan tâm đến sự phi lý này, bởi vì không có trường hợp nào trong data với trọng lượng gần 58 kg.

Tại sao mô hình bậc 6 là fit hoàn hảo được? Bởi vì nó có đủ parameter để gán cho từng điểm trong data. Phương trình của mô hình có trung bình cho 7 parameter:

$$ \mu_i = \alpha + \beta_1 m_i + \beta_2 m_i^2 + \beta_3 m_i^3 + \beta_4 m_i^4 + \beta_5 m_i^5 + \beta_6 m_i ^6 $$

và có 7 loài để dự đoán kích thước não. Cho nên để hiệu quả cao, mô hình này gán một parameter độc nhất để tái thể hiện mội giá trị quan được của data. Đây là một hiện tượng thông thường: Nếu bạn sử dụng một họ các mô hình với vừa đủ parameter, bạn có thể fit data một cách hoàn hảo. Nhưng mô hình như vậy sẽ cho dự đoán phi lý cho những trường hợp chưa quan sát được.

<div class="alert alert-info">
    <p><strong>Fit mô hình được xem như nén data.</strong>Một góc nhìn khác về mô hình phi lý như trên là xem việc fit mô hình như một dạng của <strong>NÉN DATA (DATA COMPRESSION)</strong>. Parameter tổng hợp những mối liên hệ trong data. Việc tổng kết sẽ nén data lại thành dạng đơn giản hơn, mặc dù vẫn có sự mất mát thông tin ("lossy" compression). Parameter có thể dùng để tạo data mới, giải nén data hiệu quả.</p>
    <p>Khi mô hình có mỗi parameter tương ứng với mỗi giá trị data, như <code>m7_6</code>, thì điều đó có nghĩa là không có sự nén xảy ra. Mô hình chỉ mã hoá data thô thành ở dạng khác, dùng parameter thay thế. Kết quả là, ta không học gì thêm được từ data ở mô hình như vậy. Để học từ data cần mô hình đơn giản đạt được mức độ nén nhất định nhưng không quá nhiều. Góc nhìn này ở việc chọn mô hình, thường được biết đến là <strong>MINIMUM DESCRIPTION LENGTH (MDL)</strong>.<sup><a name="r103" href="#103">103</a></sup></p>
</div>

### 7.1.2 Quá ít parameter cũng không tốt

Mô hình polynomial bị overfit fit data cực tốt, nhưng chúng vẫn bị ảnh hưởng bởi độ chính xác trong mẫu, thể hiện qua những dự đoán phi lý với data ngoài mẫu. Ngược lại, <strong>UNDERFITTING</strong> tạo ra mô hình chưa chính xác cả trong và ngoài mẫu. Chúng học quá ít, thất bại trong việc phục hồi đặc tính của mẫu.

Một cách khác để khái niệm hoá mô hình bị underfit là nhận ra rằng nó không nhạy cảm với mẫu. Chúng ta có thể loại bỏ bất kỳ một điểm nào trong mẫu và vẫn có được đường hồi quy tương tự. Ngược lại, mô hình phức tạp nhất, `m7_6`, rất nhạy cảm với mẫu. Nếu chúng ta loại bỏ bất kỳ một điểm trong mẫu, đường trung bình sẽ thay đổi rất nhiều. Bạn sẽ thấy sự nhạy cảm này ở hình dưới đây. Trong cả hai hình, tôi có bỏ từng dòng một trong data, và tính lại posterior. Ở bên trái, mỗi đường thẳng là mô hình bậc 1, `m7_1`, fit vào 7 bộ khả thi của data bằng cách bỏ từng dòng một. Những đường cong ở bên phải thuộc mô hình bậc 4, `m7_4`. Chú ý rằng đường thẳng thì ít dao động, trong khi đường cong bay tung toé. Đây là hiện tương phản thường gặp giữa mô hình bị underfit và overfit: độ nhạy với các thành phần của chính data dùng để fit mô hình.

![](/assets/images/fig 7-4.svg)
<details class="fig"><summary>Hình 7.4: Hiện tượng underfit và overfit dưới dạng kém nhạy và quá nhạy với mẫu.</summary>
<pre><code>def model1(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic("mu", a + b * mass_std)
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std) 
def model4(mass_std, brain_std=None):
    a = numpyro.sample("a", dist.Normal(0.5, 1))
    b = numpyro.sample("b", dist.Normal(0, 10).expand([4]))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0, 1))
    mu = numpyro.deterministic(
        "mu", a + jnp.sum(b * jnp.power(mass_std.reshape(-1,1), jnp.arange(1, 5)), 1)
    )
    numpyro.sample("brain_std", dist.Normal(mu, jnp.exp(log_sigma)), obs=brain_std)
fig, ax = plt.subplots(1,2, figsize=(10,5))
for a, title, md in zip(ax, ['m7_1', 'm7_4'], [model1, model4]):
    az.plot_pair(d[["mass", "brain"]].to_dict("list"),
                 ax=a, scatter_kwargs={"markersize":10, "marker":"o"})
    a.set_title(title)
    a.set_xlabel('body mass(kg)')
    a.set_ylabel('brain volume(cc)')
    for i in range(7):
        d_minus = d.drop(i)
        d_minus["mass_std"] = (d_minus.mass - d_minus.mass.mean() )/d_minus.mass.std()
        d_minus["brain_std"] = d_minus.brain / d_minus.brain.max()
        m = AutoLaplaceApproximation(md)
        svi = SVI(md, m, optim.Adam(0.01), Trace_ELBO(),
            mass_std=d_minus.mass_std.values,
            brain_std=d_minus.brain_std.values,
        )
        p, loss = svi.run(random.PRNGKey(2), 5000)
        post = m.sample_posterior(random.PRNGKey(1), p, (1000,))
        mass_seq = jnp.linspace(d_minus.mass_std.min(), d_minus.mass_std.max(), num=100)
        l = Predictive(md, post, return_sites=["mu"])(
            random.PRNGKey(2), mass_std=mass_seq
        )["mu"]
        mu = jnp.mean(l, 0)
        a.plot(mass_seq*d_minus.mass.std() +d_minus.mass.mean(), mu*d_minus.brain.max(), "k", linewidth=1)
plt.tight_layout()</code></pre></details>

<div class="alert alert-dark">
    <p><strong>Loại bỏ các dòng.</strong> Phép tính cần để tạo hình 7.4 có thể được làm dễ dàng qua hàm index trong dataframe. Để xoá dòng <code>i</code> ở dataframe <code>d</code>, chỉ cần:</p>
    <pre><code>i = 1
d_minus_i = d.drop(i)</code></pre>
    <p>Code trên sẽ loại bỏ dòng <code>i</code> và giữ tất cả các cột. Lặp lại hồi quy chỉ là vấn đề vòng lặp toàn bộ các dòng.</p>
</div>

<div class="alert alert-info">
    <p><strong>Sai lệch (bias) và phương sai (variance).</strong> Hai hiện tượng underfitting/overfitting thường được mô tả là <strong>ĐÁNH ĐỔI SAI LỆCH VÀ PHƯƠNG SAI (BIAS-VARIANCE TRADE-OFF).</strong><sup><a name="r104" href="#104">104</a></sup> Mặc dù có hơi khác biệt về hai khác niệm này, bias-variance trade-off được đặt ra cho vấn đề tương tự. "Bias" liên quan đến underfitting, trong khi "variance" liên quan đến overfitting. Những từ này dễ gây nhầm lẫn, bởi vì chúng được dùng bằng nhiều cách ở nhiều bối cảnh, ngay cả trong thống kê. Từ "bias" nghe có vẻ là một chuyện xấu, mặc dù tăng bias thường dẫn đến dự đoán tốt hơn.</p>
</div>

## <center>7.2 Entropy và độ chính xác</center><a name="a2"></a>

Vậy chúng ta phải định hướng giữa thuỷ quái overfitting và lốc xoáy underfitting? Cho dù bạn có dùng regularization hoặc tiêu chuẩn thông hoặc cả hai, việc đầu tiên bạn phải làm là chọn một tiêu chuẩn (criterion) cho hiệu năng mô hình. Bạn muốn mô hình của bạn chạy tốt như thể nào? Ta sẽ gọi tiêu chuẩn này là *mục tiêu*, và trong phần này bạn sẽ thấy thuyết thông tin sẽ cho một *mục tiêu* dễ dùng và hiệu quả.

Tuy nhiên, con đường đến độ lệch lạc ngoài mẫu (out-of-sample deviance) rất ngoằn ngoèo. Đây là những bước đầu tiên. Trước hết, chúng ta phải xác lập một con số đo khoảng cách đến độ chính xác hoàn mỹ. Điều này cần đến một chút *thuyết thông tin (information theory)*, vì nó cho một số đo tự nhiên cho khoảng cách giữa hai phân phối xác suất. Thứ hai, chúng ta phải xác lập *độ lệch lạc (deviance)* để ước lượng khoảng cách tương đối đến độ chính xác hoàn mỹ. Sau cùng, chúng ta xác lập là chính deviance ngoài mẫu là mối quan tâm chính. Một khi bạn có deviance trong tay như là thước đo hiệu năng mô hình, trong phần sau bạn sẽ thấy cả regularizing prior và tiêu chuẩn thông tin giúp chúng ta cải thiện và ước lượng deviance ngoài mẫu của mô hình.

Tài liệu của phần này rất khó. Bạn không cần phải hiểu hết ở lần đọc đầu tiên.

### 7.2.1 Đuổi việc người dự báo thời tiết.

Độ chính xác thì dựa vào định nghĩa của mục tiêu, và không có mục tiêu nào là tốt nhất. Để định nghĩa mục tiêu, có hai phương diện lớn cần phải suy nghĩ:
1. *Phân tích hao phí-lợi ích.* Nếu chúng ta sai thì hao phí là bao nhiêu? Nếu chúng ta đúng thì thắng bao nhiêu? Đa phần nhà khoa học không bao giờ đặt câu hỏi ở dạng này, nhưng nhà khoa học ứng dụng phải thường xuyên trả lời nó.
2. *Độ chính xác trong bối cảnh.* Một vài công việc dự đoán thì dễ dàng hơn những công việc khác. Cho nên ngay cả chúng ta mặc kệ hao phí và lợi ích, chúng ta cần một cách để đánh giá "độ chính xác" liên quan đến cải thiện mô hình.

Nó sẽ giúp ích nếu ta khám phá hai phương diện trên bằng một ví dụ. Giả sử trong một thành phố cụ thể, một người dự báo thời tiết thực hiện dự đoán ngày mưa hoặc ngày nắng vào mỗi ngày trong năm.<sup><a name="r105" href="#105">105</a></sup> Dự đoán ở dưới dạng tỉ lệ mưa vào ngày đó. Người dự báo thời tiết hiện đang công tác dự đoán rằng trong 10 ngày tiếp theo, kết cục sẽ như bảng sau:

| Ngày    | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---------|---|---|---|---|---|---|---|---|---|----|
|Dự đoán  | 1 | 1 | 1 |0.6|0.6|0.6|0.6|0.6|0.6|0.6 |
|Quan sát | R | R | R | \-| \-| \-| \-| \-| \-| \- |

(Ngày mưa là R, ngày nắng là -)

Một người mới vào thành phố vào tự cao rằng anh ta có thể thắng được người dự báo thời tiết hiện tại bằng cách luôn dự đoán toàn ngày nắng. Trong khoảng thời gian 10 ngày, bảng báo cáo của người mới sẽ là:

| Ngày    | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---------|---|---|---|---|---|---|---|---|---|----|
|Dự đoán  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0  |
|Quan sát | R | R | R | \-| \-| \-| \-| \-| \-| \- |

"Cho nên với một mình tần suất chính xác của dự đoán", người mới tuyên bố, "Tôi là người tốt nhất cho công việc này."

Người mới đã đúng. Định nghĩa *tần suất dự đoán trúng* như là cơ hội trung bình của dự đoán chính xác. Vậy với người dự báo thời tiết hiện tại, cô ấy có được $ 3 \times 1 + 7 \times 0.4 = 5.8$ lần trúng trong 10 ngày, với tần suất dự đoán chính xác trong một ngày là $5.8/10 =0.58$. Ngược lại, người mới có được $ 3 \times 0 + 7 \times 1 = 7$ lần trúng, với tần suất $ 7/10=0.7$  trong một ngày. Người mới đạt được chiến thắng.

#### 7.2.1.1 Hao phí và lợi ích

Nhưng không khó để tìm một tiêu chuẩn khác ngoài tần suất dự đoán đúng ra, mà có thể làm cho người mới trông ngu ngốc. Chỉ cần sự xem xét hao phí và lợi ích là đủ. Giả sử bạn ghét bị mắc mưa, nhưng bạn cũng ghét phải mang theo dù. Hãy định nghĩa rằng hao phí khi bị ướt là $-5$ điểm niềm vui và hao phí phải mang dù là $-1$ điểm niềm vui. Giả sử cơ hội bạn mang dù thì tương đương với xác suất mưa được dự báo. Công việc của bạn là tối đa hoá điểm niềm vui thông qua chọn người dự báo thời tiết. Đây là bảng điểm, khi bạn chọn một trong hai người:

| Ngày    | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---------|---|---|---|---|---|---|---|---|---|----|
|Quan sát | R | R | R | \-| \-| \-| \-| \-| \-| \- |
|Điểm     |   |   |   |   |   |   |   |   |   |    |
|Người hiện tại| -1| -1| -1|-0.6|-0.6|-0.6|-0.6|-0.6|-0.6|-0.6|
|Người mới|-5|-5|-5|0|0|0|0|0|0|0|

Cho nên người dự báo thời tiết hiện tại sẽ cho bạn $ 3 \times (-1) + 7 \times (-0.6)=-7.2$ điểm niềm vui, trong khi người mới cho $-15$ điểm niềm vui. Vậy người mới bây giờ trông có vẻ không thông minh. Bạn có thể thay đổi hao phí và luật chơi ở đây, nhưng vì người mới luôn luôn không dự đoán mưa, bạn có thể thắng anh ta dễ dàng.

#### 7.2.1.2 Đo đạc độ chính xác

Nhưng ngay nếu chúng ta bỏ qua hao phí và lợi ích của quyết định thực tế dựa vào dự báo, vẫn có nhiều yếu tố mơ hồ về phương pháp đo đạc "độ chính xác" để sử dụng. Không có gì đặc biệt về "tần suất dự đoán trúng". Câu hỏi nên tập trung vào: Định nghĩa nào của "độ chính xác" được tối đa hoá khi biết mô hình thực tạo data? Chắn chắn chúng ta không thể nào làm tốt hơn như thế.

Xem xét việc tính chính xác xác suất của dự đoán những ngày tiếp theo. Điều này có nghĩa là tính xác suất dự đoán đúng cho mỗi ngày. Sau đó nhân chúng với nhau để có được xác suất liên hợp (joint probability) của những dự đoán đúng trong chuỗi ngày tiếp theo. Điều này giống như khả năng liên hợp (joint likelihood), giống như những gì bạn đã dùng tới bây giờ. Đây là định nghĩa độ chính xác có thể được tối đa hoá bằng mô hình đúng.

Dưới ánh sáng mới này, người mới càng nhìn tệ. Xác suất của người hiện tại là $1^3 \times 0.4^7 \approx 0.005$. Còn người mới là $0^3 \times 1^7 =0$. Vậy người mới có zero xác suất để có được chuỗi dự đoán đúng. Đó là bởi vì dự đoán của người mới không bao giờ có mưa. Cho nên ngay cả người mới có xác suất chính xác trung bình (tần suất dự đoán trúng) cao hơn, anh ta có xác suất liên hợp của độ chính xác rất tệ hại.

Và xác suất liên hợp là thước đo chúng ta cần. Tại sao? Bởi vì nó xuất hiện trong Bayes' theorem như trong likelihood. Đó là một thước đo duy nhất cho phép đếm chính xác toàn bộ các cách tương đối mà mỗi sự kiện (chuỗi ngày mưa và ngày nắng) có thể xảy ra. Một định nghĩa khác là xem xét những gì xảy ra khi hcúng ta tối đa hoá xác suất trung bình của xác suất liên hợp. Mô hình thực tế tạo data sẽ không có tần suất dự đoán trúng tốt nhất. Bạn đã thấy điều này ở ví dụ dự báo thời tiết: Gán xác suất zero cho này mưa sẽ tăng tần suất trúng, nhưng nó sai rõ ràng. Ngược lại, mô hình đúng sẽ có xác suất liên hợp cao nhất.

Trong tài liệu thống kê, bạn sẽ hay gặp số đo chính xác này được gọi là **LUẬT TÍNH ĐIỂM LOG (LOG SCORING RULE)**, bởi vì theo kinh điển bạn phải tính log của xác suất liên hợp và báo cáo nó. Nếu bạn thấy một phân tích dùng một thang điểm khác, nó có thể là trường hợp đặc biệt của luật tính điểm log hoặc một thứ khác tệ hơn.

<div class="alert alert-info">
    <p><strong>Hiệu chuẩn được sử đúng quá đà.</strong> Thông thường các mô hình được đánh giá trong qua <strong>TRỊ SỐ HIỆU CHUẨN (CALIBRATION)</strong>. Nếu mô hình dự đoán xác suất ngày mưa là 40%, thì nó được cho là đã được "hiệu chuẩn", nếu trên thực tế ngày mưa xảy ra 40% của dự đoán đó. Vấn đề là dự đoán được hiệu chính chưa chắc gì là tốt. Ví dụ, nếu nó mưa 40% số ngày, thì mô hình chỉ dự đoán 40% ngày mưa vào tất cả các ngày sẽ là mô hình hiệu chuẩn tốt nhất. Nhưng nó cũng sẽ cho dự đoán tệ hại. Dự đoán tốt không cần sự hiệu chuẩn. Giả sử người dự báo luôn có 100% tự tin rằng anh ta luôn dự đoán đúng 80% các ngày. Người dự báo là chính xác, nhưng anh ta không có sự hiệu chuẩn. Anh ta là tự tin quá mức.</p>
    <p>Đây là một ví dụ thực. Trang web dự báo thời tiết www.fivethirtyeight.com cho rất nhiều dự đoán. Trong đó sự hiệu chuẩn cho sự kiện thể thao là gần như hoàn hảo.<sup><a name="r106" href="#106">106</a></sup> Nhưng độ chính xác của họ chỉ nhỉnh hơn đoán mò một chút. Ngược lại, dự đoán chính trị của nó thì ít hiệu chuẩn hơn, nhưng trung bình của độ chính xác thì cao hơn.</p>
    <p>Từ ngữ như "hiệu chuẩn" có rất nhiều ý nghĩa. Cho nên đó là điều tốt để cung cấp và yêu cầu các dữ kiện bối cảnh.<sup><a name="r107" href="#107">107</a></sup> Việc kiểm tra dự đoán posterior được dùng trong sách này, đôi khi được gọi là "kiểm tra hiệu chuẩn (calibration check)".</p>
</div>

### 7.2.2 Thông tin và tính bất định

Chúng ta muốn dùng logarith xác suất (log prob) của data để cho điểm độ chính xác của nhưng mô hình được chọn. Vấn đề tiếp theo là làm cách nào để đánh giá khoảng cách đến dự đoán chính xác nhất - dự đoán hoàn mỹ. Dự đoán hoàn mỹ sẽ cho xác suất thực của mưa vào mỗi ngày. Vậy khi cả hai người dự báo thời tiết cung cấp dự đoán khác nhau đến mục tiêu, ta có đo khoảng cách của hai dự đoán đến mục tiêu. Nhưng khoảng cách nào ta sẽ áp dụng? Không rõ làm thế nào để trả lời câu hỏi này. Nhưng thực ra chỉ có một câu trả lời độc nhất và tối ưu nhất.

Để có được đáp án cần phải biết rằng ta phải làm gì với điểm chính xác. Vài mục tiêu sẽ dễ trúng hơn mục tiêu khác. Ví dụ, giả sử ta kéo dài dự báo thời tiết sang mùa đông. Bây giờ có ba loại ngày: mưa, nắng, tuyết. Bây giờ có ba cách để cho đáp án sai, thay vì hai. Điều này phải được phản ảnh trong thang đo khoảng cách đến mục tiêu, bởi vì thêm một loại sự kiện mới, mục tiêu sẽ khó trúng hơn.

Nó giống như phải nhắm vào bia bắn cung hai chiều và bắt cung thủ bắn trúng mục tiêu vào đúng thời điểm - chiều thứ ba. Bây giờ khoảng cách có thể giữa cung thủ giỏi nhất và cung thủ tệ nhất đã kéo dài ra, bởi vì có thêm cách khác để bắn hụt. Và với việc cách bắn hụt mới, anh có thể nói rằng có thêm cách mới để cung thủ gây ấn tượng. Vậy khoảng cách tiềm năng giữa mục tiêu và mũi tên tăng thêm, cũng như tiềm năng cải thiện và khả năng của cung thủ giỏi để gây ấn tượng.

Giải pháp cho vấn đề đo khoảng cách giữa độ chính xác mô hình đến mục tiêu, được sáng tạo ra vào cuối những năm 1940.<sup><a name="r108" href="#108">108</a></sup> Bắt nguồn từ việc ứng dụng ở các vấn đề trong giao tiếp tin nhắn, như điện báo, lĩnh vực **THUYẾT THÔNG TIN (INFORMATION THEORY)** bây giờ rất quan trọng trong khoa học cơ bản và ứng dụng, và nó có quan hệ mật thiết với suy luận Bayes. Và cũng giống như nhiều lĩnh vực thành công khác, thuyết thông tin đã tạo ra nhiều ứng dụng không có thật.<sup><a name="r109" href="#109">109</a></sup>

Một ý niệm cơ bản là đặt câu hỏi: *Tính bất định của mô hình sẽ giảm đi bao nhiêu sau khi biết được kết cục?* Hãy xem lại ví dụ dự báo thời tiết. Dự báo được đưa ra nhưng thời tiết thì chưa biết. Khi ngày đó đến thực sự, thời tiết sẽ không còn bất định nữa. Mức độ giảm xuống của tính bất định là một số đo tự nhiên về chúng ta đã học được bao nhiêu, mức độ "thông tin" mà ta suy được từ việc quan sát kết cục. Vậy nếu ta có thể phát triển định nghĩa chính xác của "tính bất định (uncertainty)", ta sẽ cung cấp được số đo nền về dự đoán khó đến mức nào, cũng như sự cải thiện sẽ thêm bao nhiêu. Con số đo mức độ giảm của tính bất định là định nghĩa của *thông tin* trong bối cảnh này.

>*Thông tin*: mức độ giảm của tính bất định khi chúng ta biết kết cục.

Để dùng định nghĩa này, ta cần một nguyên tắc để định lượng tính bất định tồn tại trong một phân phối xác suất. Giả sử có hai sự kiện thời tiết khả thi vào một ngày cụ thể: Hoặc là nắng hoặc là mưa. Mỗi sự kiện xảy ra với một xác suất, và tổng xác suất là một. Chúng ta cần một hàm số để dùng xác suất của ngày nắng và ngày mưa vào tạo ra con số của tính bất định.

Có rất nhiều cách để đo đạc tính bất định. Cách thông dụng nhất bắt đầu bằng việc đặt tên cho vài thuộc tính của giá trị đo lường tính bất định. Có ba thuộc trực quan:
1. Con số đo lường tính bất định phải là số liên tục. Nếu không, một thay đổi nhỏ trong xác suất, ví dụ như xác suất ngày mưa, có thể cho sự thay đổi lớn của tính bất định.
2. Con số đo lường tính bất định phải tăng lên khi số lượng sự kiện khả thi tăng lên. Ví dụ, có hai thành phố cần dự báo thời tiết. Ở thành phố thứ nhất, nó mưa vào một nửa các ngày trong năm và ngày nắng những ngày còn lại. Ở thành phố thứ hai, có ngày mưa, ngày nắng, ngày tuyết, lần lượt mỗi 3 ngày trong năm. Chúng ta muốn số đo tính bất định
phải lớn hơn ở thành phố thứ hai, nơi có nhiều loại sự kiện hơn để dự báo.
3. Con số đo lường tính bất định phải là từ phép cộng. Nghĩa là chúng ta trước tiên đo tính bất định ngày mưa hoặc ngày nắng (2 loại sự kiện) và sau đó tính bất định của nóng và lạnh (2 loại sự kiện khác), tính bất định của cả 4 cặp của những sự kiện này - mưa/nóng, mưa/lạnh, nắng/nóng, nắng/lạnh - phải là tổng của từng tính bất định một.

Chỉ có một hàm số thoả mãn các thuộc tính này. Hàm số này thường được biết đến với tên gọi **ENTROPY THÔNG TIN (INFORMATION ENTROPY)**, và nó có định nghĩa đơn giản đến ngạc nhiên. Nếu có $n$ sự kiện khả thi khác nhau và mỗi sự kiện $i$ có xác suất $p_i$, và ta gọi danh sách các xác suất là $p$, thì số đo tính bất định ta cần tìm là:

$$H(p) = - E \log(p_i) = - \displaystyle \sum_{i=1}^{n} p_i \log(p_i) \quad (7.1)$$

Hay ta có thể đọc thành:

>Tính bất định có ở phân phối xác suất là logarith xác suất trung bình của sự kiện.

"Sự kiện" ở đây có thể ám chỉ một loại thời tiết, như ngày nắng và ngày mưa, hoặc một loại chim, hoặc thậm chí một nucleotide cụ thể trong trình tự DNA.

Trong khi ta không cần biết chi tiết nguồn gốc của $H$, ta cần biết rằng không có sự ngẫu nhiên nào trong hàm số này. Tất cả các thành phần của nó đều xuất phát từ ba thuộc tính nói trên. Tuy thế, ta chấp nhận $H(p)$ là số đo tính bất định hữu ích không phải bởi vì những tiền đề dẫn đến nó, mà nó thực sự là một công cụ hữu ích và ứng dụng cao.

Một ví dụ sẽ giúp xoá tan những hoang mang của hàm số $H(p$). Để tính entropy thông tin cho dự báo thời tiết, giả sử xác suất thực của ngày mưa và ngày nắng lần lượt là $p_1=0.3$ và $p_2=0.7$. Thì:

$$ H(p) = - \big(p_1 \log(p_1) +p_2 \log(p_2)\big) \approx 0.61 $$

```python
p = jnp.array([0.3, 0.7])
-jnp.sum(p * jnp.log(p))
```
<samp>0.6108643</samp>

Giả sử chúng ta sống ở Abu Dhabi. Thì xác suất ngày mưa và ngày nắng sẽ trông giống như $p_1=0.01$ và $p_2=0.99$. Bây giờ Entropy sẽ vào khoảng 0.06. Tại sao nó lại giảm xuống? Bởi vì ở Abu Dhabi rất hiếm xảy ra mưa. Cho nên có ít tính bất định hơn vào một ngày bất kỳ, so với nơi mà mưa 30% số ngày. Nó theo cách mà entropy thông tin đo lường tính bất định nằm trong một phân phối các sự kiện. Tương tự, nếu chúng ta thêm một loại sự kiện khác vào phân phối - dự báo vào mùa đông, nên dự đoán thêm ngày tuyết - entropy có xu hướng tăng, do có chiều không gian mới vào vấn đề dự đoán. Ví dụ, giả sử xác suất ngày mưa, nắng, tuyết lần lượt là $p_1 =0.7$, $p_2=0.15$, và $p_3=0.15$. Thì entropy khoảng 0.82.

Những giá trị entropy này một mình nó không có ý nghĩa gì đối với chúng ta. Mà chúng ta có thể dùng nó để  phần kế tiếp.

<div class="alert alert-dark">
    <p><strong>Vài điều về entropy.</strong> Như tôi đã nói về entropy thông tin là logarith xác suất trung bình. Nhưng vẫn còn con số -1 trong định nghĩa. Nhân log prob với -1 sẽ làm entropy tăng thêm từ zero, hơn là giảm từ zero. Đây là do sự tiện lợi, không mang tính chức năng. Logarith trên là log tự nhiên (cơ số $e$), nhưng thay đổi cơ số cũng không ảnh hưởng gì đến suy luận. Log cơ số 2 cũng thường gặp. Chỉ cần tất cả entropy bạn so sánh dùng chung cơ số, bạn sẽ ổn.</p>
    <p>Khó khăn ở đây để tính $H$ là câu hỏi không tránh được khi ta gặp $p_i=0$, ta sẽ làm gì?. Log(0) = $-\infty$, ta không dùng được nó. Tuy nhiên, định luật L'Hôpital nói rằng $\lim_{p_i \to 0} p_i \log(p_i) =0$. Cho nên ta có thể giả định rằng $0 \log(0) =0$, khi ta tính $H$. Hay nói cách khác, những sự kiện không bao giờ xảy ra thì được loại bỏ. Cần nhớ rằng khi một sự kiện không bao giờ xảy ra, ta không có lý do gì giữ nó trong mô hình.</p>
</div>

<div class="alert alert-info">
    <p><strong>Lợi ích của tối đa hoá tính bất định.</strong> Thuyết thông tin có rất nhiều ứng dụng. Một ứng dụng đặc biệt quan trọng là <strong>MAXIMUM ENTROPY</strong>, hay còn gọi là <strong>MAXENT</strong>. Maxent là một nhánh kỹ năng dùng để tìm ra phân phối xác suất mà hằng định với tình trạng kiến thức. Nói cách khác, với những gì chúng ta biết, thì phân phối nào là <i>ít ngạc nhiên</i> nhất? Thực vậy, trả lời cho câu hỏi này là tối đa hoá entropy thông tin, sử dụng prior làm ràng buộc (constrain).<sup><a name="r110" href="#110">110</a></sup> Nếu bạn dùng nó, bạn sẽ có được phân phối posterior. Cho nên cập nhật Bayes chính là tối đa hoá entropy. Ở Chương 10, maximum entropy sẽ giúp ta xây dựng mô hình tuyến tính tổng quát (GLM).</p>
</div>

### 7.2.3 Từ entropy đến độ chính xác

Thật là tốt khi có cách định lượng tính bất định. $H$ cho phép ta điều đó. Nên chúng ta có thể nói rằng, một cách chính xác, độ khó của việc dự đoán trúng mục tiêu. Nhưng làm sao để sử dụng entropy thông tin để nói rằng mô hình cách xa mục tiêu bao nhiêu? Chìa khoá nằm ở **ĐỘ PHÂN KỲ (DIVERGENCE)**:

>Độ phân kỳ: tính bất định thêm vào khi sử dụng xác suất từ phân phối này để mô tả phân phối kia.

Đây thường được gọi là *Kullback-Leiber divergence* hay đơn giản là độ phân kỳ KL (KL divergence), được đặt tên theo người tạo ra nó.<sup><a name="r111" href="#111">111</a></sup>

Giả sử phân phối thực của sự kiện là $p_1 =0.3$, $p_2=0.7$. Nếu chúng ta tin rằng xác suất của những sự kiện này thay vào đó là $q_1 =0.25$, $q_2=0.75$, ta đã tăng tính bất định thêm vào là bao nhiêu, sau hệ quả ta dùng $q=\{q_1, q_2\}$ để ước lượng $p=\{p_1, P-2\}$? Đáp án chuẩn của câu hỏi này này dựa vào $H$, và có công thức cũng đơn giản:

$$D_{KL}(p,q) = \displaystyle\sum_i p_i \big(\log(p_i)-\log(q_i)\big) = \displaystyle\sum_i p_i \log \left( \frac{p_i}{q_i} \right) $$

Nếu dùng văn nói thì độ phân kỳ là *trung bình của hiệu giữa logarith xác suất của mục tiêu ($p$) và mô hình ($q$).* Độ phân kỳ chỉ là hiệu của hai entropy: Entropy của phân phối mục tiêu $p$ và *cross entropy* xuất hiện từ việc dùng $q$ để dự đoán $p$. Khi $p=q$, chúng ta biết xác suất thực của những sự kiện. Khi đó:

$$D_{KL}(p,q) = D_{KL}(p,p) = \displaystyle\sum_i p_i \big(\log(p_i)-\log(p_i)\big) = 0$$

Không có tính bất định thêm vào khi ta dùng bản thân phân phối xác suất để tự đại diện nó. Nó là một suy nghĩ dễ chịu.

Nhưng quan trọng hơn, khi $q$ tăng lên khác nhiều với $p$, độ phân kỳ $D_{KL}$ cũng tăng. Hình sau đây là một ví dụ. Giả sử phân phối mục tiêu thật $p=\{0.3, 0.7\}$. Giả sử phân phối dùng để ước lượng là $q$, có thể là bất kỳ giá trị nào trong khoảng $q=\{0.01, 0.99\} $ đến $q=\{0.99, 0.01\}$. Xác suất đầu tiên, $q_1$, được biểu diễn ở trục hoành, và trục tung biểu diễn $D_{KL}(p,q)$. Khi và chỉ khi $q=p$, ở $q_1=0.3$, thì độ phân kỳ đạt được giá trị zero. Mọi điểm khác, nó đều tăng.

![](/assets/images/fig 7-5.svg)
<details class="fig"><summary>Hình 7.5: Độ phân kỳ (divergence) thông tin của phân phối $q$ để ước lượng phân phối thực $p$. Độ phân kỳ chỉ bằng zero khi $p=q$ (đường nét đứt). Ngược lại, độ phân kỳ là dương và ngày càng tăng khi $q$ trở nên khác xa với $p$. Khi chúng ta có hơn một phân phối ứng cử $q$ khác, phân phối $q$ có độ phân kỳ nhỏ nhất là ước lượng chính xác nhất, theo một cách nói khác là có tính bất định thêm vào ít nhất.</summary>
<pre><code>def DKL(p,q):
    return jnp.sum(p * jnp.log(p/q), axis=1)
q1 = jnp.arange(0.01, 1, 0.01)
q = jnp.array([q1, 1-q1]).T
dkl_val = DKL(p,q)
q_best = q1[dkl_val.argmin()]

plt.figure(figsize=(5,5))
plt.plot(q1, dkl_val)
plt.vlines(q_best,0,dkl_val.max(), linestyles="dashed", linewidth=1)
plt.ylabel("Độ phân kỳ của $q$ đến $p$")
plt.xlabel("$q_1$")
plt.annotate("$q=p$", (q_best+0.02, 1.5))</code></pre></details>

Độ phân kỳ có thể giúp chúng ta thể hiện sự tương phản giữa nhiều mô hình khác nhau ước lượng cho $p$. Khi mà hàm số $q$ cho ước lượng chính xác hơn, $D_{KL}(p,q)$ sẽ thu nhỏ lại. Cho nên nếu chúng ta có một cặp phân phối được ứng cử, thì ứng cử viên có độ phân kỳ nhỏ hơn hơn sẽ gần mục tiêu hơn. Bởi vì mô hình dự đoán cụ thể hoá xác suất của các sự kiện (quan sát), chúng ta có thể dùng độ phân kỳ để so sánh độ chính xác của mô hình.

<div class="alert alert-dark">
    <p><strong>Cross entropy và độ phân kỳ.</strong> Việc suy ra công công thức độ phân kỳ dễ hơn bạn nghĩ. Ý tưởng ở đây là nhận ra khi chúng ta dùng phân phối xác suất $q$ để dự đoán sự kiện từ phân phối khác $p$, thì đó là định nghĩa của <i>cross entropy</i>: $H(p,q) = -\sum_i p_i \log(q_i)$. Ký hiệu này nghĩa là sự kiện xuất phát từ $p$, nhưng lại được mong đợi rằng cho ra $q$, nên entropy bị dồn lại, phụ thuộc vào sự khác nhau giữa $p$ và $q$. Độ phân kỳ được định nghĩa là phần entropy <i>thêm vào</i> tạo ra khi dùng $q$. Cho nên nó thực chất là hiệu giữa $H(p)$, entropy thực của sự kiện, và $H(p,q)$:
        $$\begin{aligned}
        D_{KL}(p,q) &= H(p,q) - H(p)\\
        &= - \displaystyle\sum_i p_i \log(q_i) - \big( - \displaystyle\sum_i p_i \log(p_i)\big) = - \displaystyle\sum_i p_i \big( \log(q_i) - \log(p_i)\big)\\
        \end{aligned}$$
    Cho nên độ phân kỳ thực ra là đo lường $q$ khác xa $p$ như thế nào, với đơn vì là entropy. Chú ý rằng mục tiêu khá quan trọng: $H(p,q)$ nói chung không bằng $H(q,p)$. Để rõ hơn, mời bạn xem phần dưới.</p>
</div>

<div class="alert alert-info">
    <p><strong>Độ phân kỳ phụ thuộc vào chiều hướng.</strong> Nói chung, $H(p,q)$ không bằng $H(q,p)$. Chiều hướng ảnh hưởng đến phép tính độ phân kỳ. Hiểu được tại sao có điều này cũng có ích, và sau đây là một ví dụ.</p>
    <p>Giả sử ta du hành vào Sao Hoả. Nhưng chúng ta không thể kiểm soát được điểm rơi của phi thuyền khi chúng ta đến Sao Hoả. Hãy thử dự đoán xem chúng ta sẽ rơi vào đất liền hay biển, sử dụng Trái Đất làm phân phối xác suất $q$, để ước lượng phân phối thực của Sao Hoả, $p$. Trái Đất là $q= \{0.7, 0.3\}$, lần lượt cho xác suất biển và đất liền. Sao Hoả thì rất khô, nhưng để làm ví dụ thì nó có khoảng 1% bề mặt biển, cho nên $p=\{0.01, 0.99\}$. Nếu chúng ta đếm cả các tảng băng, thì cũng không hề hấn gì. Bây giờ ta tính độ phân kỳ từ Trái Đất đến Sao Hoả. Kết quả là $D_{E\to M} = D_{KL}(p,q) =1.14$. Nó là tính bất định thêm vào cho sử dụng Trái Đất để dự đoán điểm rơi của Sao Hoả. Bây giờ ta xem xét ở chiều ngược lại. Con số trong $p$ và $q$ vẫn như nguyên, nhưng nếu ta thay đổi vai trò của chúng, thì $D_{M\to E} = D_{KL}(p,q) =2.62$. Độ phân kỳ nhiều gấp đôi ở chiều này. Kết quả này trông có vẻ khó hiểu hơn. Tại sao khoảng cách từ Trái Đất đến Sao Hoả lại ngắn hơn khoảng cách từ Sao Hoả đến Trái Đất?</p>
    <p>Đây là một đặc tính của độ phân kỳ, chứ không phải lỗi. Tính phân định thêm vào có nhiều hơn khi dùng Sao Hoả để dự đoán Trái Đất, so với dùng Trái Đất để dự đoán Sao Hoả. Lý do là, đi từ Sao Hoả đến Trái Đất, Sao Hoả có quá ít biển trên bề mặt dẫn đến chúng ta sẽ bị rất rất ngạc nhiên khi chúng ta rơi vào mặt biển trên Trái Đất. Ngược lại, Trái Đất có một lượng lớn mặt biển và đất liền. Khi chúng ta dùng Trái Đất để dự đoán Sao Hoả, chúng ta có thể dự kiến rơi xuống cả mặt biển hoặc đất liền, ở một mức nào đó, ngay cả khi chúng ta dự kiến nhiều biển hơn đất. Cho nên chúng ta sẽ không bị ngạc nhiên lắm nếu chúng ta bắt buộc rơi vào vùng đất liền trên Sao Hoả, bởi vì 30% bề mặt Trái Đất là đất liền.</p>
    <p>Một hệ quả quan trọng của tính bất đối xứng này, ở bối cảnh fit mô hình, là nếu chúng ta dùng một phân phối có entropy cao để ước lượng một phân phối thực của các sự kiện, chúng ta sẽ giảm được khoảng cách đến sự thật và theo sau đó là các sai lệch (error). Điều này sẽ giúp ta xây dựng những mô hình tuyến tính tổng quát (generalized linear model), ở Chương 10.</p>
</div>

### 7.2.4 Ước lượng độ phân kỳ

Tại thời điểm này, chắc hẳn bạn đọc sẽ thắc mắc chúng ta đang đi tới đâu. Ban đầu, mục đích là đối diện với overfitting và underfitting. Nhưng bây giờ chúng ta đã tốn nhiều giấy mực về entropy và những ảo mộng. Nó giống như tôi hứa với các bạn một ngày đi biển, rồi cuối cùng bạn lạc trong khu rừng tăm tối, tự hỏi đây có phải là đi lệch hướng hay là một âm mưu nào không.

Đây là một chuyến đi lệch hướng có chủ đích. Điểm chính của tất cả nội dung phần trước về thuyết thông tin và độ phân kỳ là để xác lập hai điều:
1. Cách đo lường khoảng cách từ mô hình đến mục tiêu của chúng ta. Thuyết thông tin cho phép chúng ta thực hiện điều đó qua độ phân kỳ KL.
2. Cách ước lượng độ phân kỳ. Thông qua việc xác định được đúng cách đo lường khoảng cách, chúng ta bây giờ cần một phương pháp ước lượng trong quy trình thiết kế mô hình thống kê thực tế.

Ta đã đạt được mục thứ nhất. Mục thứ hai là phần còn lại. Bạn bây giờ sẽ thấy được rằng độ phân kỳ cho phép dùng một đơn vị *độ lệch lạc (deviance)* để đo mức độ fit của mô hình.

Muốn dùng $D_{KL}$ để so sánh mô hình, có vẻ trước tiên ta cần phải có $p$, phân phối xác suất của mục tiêu. Trong những ví dụ nãy giờ, tôi giả định $p$ coi như là đã biết. Nhưng khi chúng ta muốn tìm một mô hình $q$ là mô hình ước lượng tốt nhất cho *sự thật* $p$, ta không có cách nào khác để tiếp cận trực tiếp $p$. Chúng ta chắc chắn không dùng đến thống kê nếu đã biết $p$.

Nhưng có một phương pháp tuyệt vời để thoát khỏi tình huống trăn trở này. Thứ chúng ta quan tâm là so sánh độ phân kỳ của những ứng cử viên khác nhau, như $q$ và $r$. Trong trường hợp đó, đa số $p$ bị tiêu biến, bởi vì có $E \log(p_i)$ trong công thức độ phân kỳ ra $q$ và $r$. Đại lượng này không ảnh hưởng đến khoảng cách từ $q$ và $r$ đến nhau. Cho nên trong khi ta không biết $p$ ở đâu, chúng ta có thể ước lượng khoảng cách giữa $q$ và $r$, và cái nào gần mục tiêu hơn. Nó giống như chúng ta không thể đoán được một cung thủ xa bảng bia cỡ nào, nhưng có thể đoán được cung thủ nào gần hơn và gần bao nhiêu.

Tất cả những điều này nghĩa là chúng ta chỉ cần logarith xác suất trung bình của mô hình: $E\log(q_i)$ cho $q$ và $E\log(r_i)$ cho $r$. Công thức này khá giống với logarith xác suất của biến kết cục mà bạn dùng để mô phỏng dự đoán của một mô hình đã fit. Thực vậy, bằng cách tính tổng các logarith xác suất của mỗi trường hợp quan sát cho phép ước lượng $E\log(q_i)$. Chúng ta không cần biết đến $p$ trong công thức tìm *mong đợi* ($E$).

Vậy chúng ta có thể so sánh logarith xác suất trung bình của mỗi mô hình để ước lượng khoảng cách tương đối từ mỗi mô hình đến mục tiêu. Điều này cũng có nghĩa là ta không diễn giải được biên độ tuyệt đối của những giá trị này - giá trị $E\log(q_i) $ hoặc $E\log(r_i)$ đều không nói lên được mô hình là tốt hay xấu. Chỉ khi có hiệu của chúng $E\log(q_i) - E\log(r_i)$ mới cho biết độ phân kỳ từ mỗi mô hình đến mục tiêu $p$.

Để tóm chúng lại vào thực hành, ta có thể làm gọn hơn bằng cách lấy tổng toàn bộ mẫu quan sát $i$, cho kết quả của mô hình $q$:

$$ S(q) = \displaystyle\sum_i\log(q_i) $$

Điểm số này còn gọi là điểm logarith xác suất (log-prob score), và nó là tiêu chuẩn vàng để so sánh độ chính xác của dự đoán của từng mô hình khác nhau. Nó cũng là ước lượng cho $E\log(q_i)$, chỉ khác ở chỗ không lấy trung bình bằng phép chia tổng số lượng mẫu.

Để tính điểm số này cho mô hình Bayes, chúng ta dùng toàn bộ phân phối posterior. Nếu không, những thiên thần sẽ nổi giận và trừng phạt bạn. Tại sao? Nếu ta không dùng toàn bộ posterior, chúng ta đang vứt bỏ thông tin. Bởi vì parameter có phân phối, cho nên dự đoán cũng có phân phối. Làm thế nào để dùng toàn bộ phân phối dự đoán? Ta cần tìm logarith của xác suất trung bình của mỗi quan sát $i$, và tính trung bình trên toàn bộ posterior. Để tính nó cần phải dùng những phép tính chính xác. Ta có thể dùng hàm `lppd` - **LOG POINTWISE PREDICTIVE DENSITY**. Ta sẽ tính `lppd` cho mô hình đầu tiên:

```python
def lppd_fn(seed, quad, params, num_samples=1000):
    post = quad.sample_posterior(random.PRNGKey(seed), params, (num_samples,))
    logprob = log_likelihood(quad.model, post, d.mass_std.values, d.brain_std.values)
    logprob = logprob["brain_std"]
    return logsumexp(logprob, 0) - jnp.log(logprob.shape[0])


lppd_fn(1, m7_1, p7_1, int(1e4))
```
<samp>0.6098668 0.6483438 0.5496093 0.6234934 0.4648143 0.4347605 -0.8444633</samp>

Mỗi giá trị này là điểm log-prob cho từng mẫu quan sát. Nhớ lại rằng có 7 mẫu quan sát trong data. Nếu chúng ta cộng chúng lại, ta sẽ có đươc tổng điểm log-prob cho mô hình và data. Những giá trị đó có ý nghĩa gì? Số lớn hơn thì tốt hơn, bởi vì có độ chính xác trung bình cao hơn. Một đại lượng khác thường gặp hơn là **ĐỘ LỆCH LẠC (DEVIANCE)**, nó cũng là điểm lppd, nhưng được nhân với $ - 2$ để cho số càng nhỏ thì càng tốt. Con số 2 ở đó là do yếu tố lịch sử.<sup><a name="r112" href="#112">112</a></sup>

<div class="alert alert-dark">
    <p><strong>Tính lppd.</strong> Phiên bản Bayes của điểm log-prob gọi là <strong>LOG-POINTWISE-PREDICTIVE-DENSITY</strong>. Với mẫu data $y$ và phân phối posterior $\Theta$:
        $$ \text{lppd}(y, \Theta) = \displaystyle\sum_i \log \frac{1}{S} \displaystyle\sum_sp(y_i | \Theta_s)$$
    Với S là số lượng trong mẫu và $\Theta_s$ là tập mẫu các parameter posterior thứ $s$. Trong khi về mặt nguyên tắc thì dễ - bạn chỉ cần tính mật độ xác suất của mỗi quan sát $i$ cho mỗi tập parameter $s$, lấy trung bình, và tính logarith - trong thực hành nó không đơn giản. Lý do là các phép tính trong máy tính thường cần một vài kỹ thuật để duy trì độ chính xác của số thập phân. Trong tính toán xác suất, ta thường tính ở thang đo logarithm để phép tính được an toàn. Đây là đoạn mã ta cần, để lặp lại phép tính ở phần trên:
    <pre><code>post = m7_1.sample_posterior(random.PRNGKey(1), p7_1, (int(1e4),))
logprob = log_likelihood(m7_1.model, post, d.mass_std.values, d.brain_std.values)
logprob = logprob["brain_std"]
n = logprob.shape[1]
ns = logprob.shape[0]
f = lambda i: logsumexp(logprob[:, i]) - jnp.log(ns)
lppd = vmap(f)(jnp.arange(n))
lppd</code></pre>
    Bạn sẽ thấy lại kết quả như trên. Đoạn mã này trước tiên tính log-prob của từng quan sát, tương tự như trong Chương 4. Nó trả kết quả là một ma trận cho mỗi quan sát $i$ và mỗi tập posterior $s$.  Hàm <code>logsumexp</code> tính logarith của tổng của $e$ mũ các giá trị đó. Tức là lấy toàn bộ log-prob của một quan sát, $e$ mũ chúng lên, cộng lại, và lấy logarith. Và phép tính này được thực hiện một cách ổn định về mặt số học. Cuối cùng trong hàm số <code>f</code> là phép trừ cho logarith của số lượng mẫu, trong trường hợp nó tương đương với chi tổng cho số lượng mẫu.</p>
</div>

### 7.2.5 Tính điểm trên data đúng

Điểm log-prob là một nguyên tắc để đo đạc khoảng cách từ mục tiêu. Nhưng điểm số tính ra được như phần trước cũng có điểm yếu như $R^2$: Nó luôn cải thiện khi mô hình phức tạp lên, ít ra với những mô hình chúng ta đã sử dụng. Giống như $R^2$, log-prob ở data dùng để huấn luyện (từ nay gọi là training data) là một giá trị đo độ chính xác dự đoán ngược lại vào data đó, không phải độ chính xác của dự đoán. Hãy tính điểm log-prob cho từng mô hình ở phần trước đầu chương:

```python
[
    jnp.sum(lppd_fn(random.PRNGKey(1), m[0], m[1])).item()
    for m in (
        (m7_1, p7_1),
        (m7_2, p7_2),
        (m7_3, p7_3),
        (m7_4, p7_4),
        (m7_5, p7_5),
        (m7_6, p7_6),
    )
]
```

<samp>2.490390 2.565982 3.695910 5.380871 14.089261 39.445390</samp>

Mô hình càng phức tạp sẽ có điểm lớn hơn! Nhưng chúng ta đã biết chúng là phi lý. Chúng ta không thể đơn thuần đánh điểm mô hình bằng năng lực của chúng ở training data. Phương pháp này sẽ đưa ta đến quái thú Scylla, kẻ huỷ diệt nhà khoa học ngây thơ.

Thực ra chúng ta cần nhất là điểm log-prob của data mới. Cho nên trước khi nhìn vào những công cụ cải thiện và đo lường điểm số mẫu ngoài data, ta hãy xem rõ vấn đề này bằng cách mô phỏng điểm số ứng với trong và ngoài mẫu. Khi chúng ta thường thu thập data và dùng chúng để fit mô hình thống kê, data đó gọi là **MẪU HUẤN LUYỆN (TRAINING SAMPLE)**. Parameter được ước lượng từ nó, và sau đó chúng ta có thể tưởng tượng sử dụng những giá trị ước lượng này để dự đoán kết cục của một mẫu mới, gọi là **MẪU KIỂM TRA (TEST SAMPLE)**. Phần mềm sẽ giúp hết cho bạn. Nhưng đây là toàn bộ quy trình, được tóm gọn lại:
1. Giả sử có mẫu huấn luyện với kích thước $N$.
2. Tính phân phối posterior của mô hình cho mẫu huấn luyện, tính điểm cho nó dựa trên mẫu huấn luyện. Gọi là $D_\text{train}$.
3. Giả sử cũng có mẫu kiểm tra cùng mô hình xử lý, kích thước $N$.
4. Tính điểm dựa trên mẫu kiểm tra, dùng posterior được tạo từ mẫu huấn luyện. Gọi đây là $D_\text{test}$.

Đây là một thí nghiệm đáng có. Nó cho phép ta khám phá sự khác nhau giữa độ chính xác đo được ứng với trong và ngoài mẫu, sử dụng tình huống dự đoán đơn giản.

Để biểu diễn đồ thị kết quả của thí nghiệm này, những gì chúng ta cần là lặp lại thí nghiệm 10000 lần, cho 5 mô hình hồi quy tuyến tính khác nhau.

Mô hình xử lý để tạo data như sau:

$$ \begin{aligned}
y_i &\sim \text{Normal}(\mu_i, 1)\\
\mu_i &= (0.15)x_{1,i} - (0.4)x_{2,i}\\
\end{aligned}$$

Mô hình tương ứng với kết cục $y$ là Gaussian với intercept $\alpha=0$, slope của mỗi biến dự đoán (predictor) là $\beta_1=0.15$, và $\beta_2=-0.4$. Mô hình phân tích data là hồi quy tuyến tính giữa 1 và 5 tham số (parameter). Mô hình đầu tiên, với 1 parameter để ước lượng, là mô hình hồi quy tuyến tính với trung bình chưa biết và $\sigma$ cố định = 1. Mỗi parameter thêm vào mô hình thêm vào mô hình thêm một predictor và hệ số beta của nó. Bởi vì mô hình "thực" có hệ số không phải zero chỉ cho 2 predictor đầu tiên, ta có thể mô hình thực có 3 parameter. Bằng cách fit cả 5 mô hình, giữa 1 và 5 parameter, với mẫu huấn luyện từ chung một nguồn gốc, chúng ta có thể thấy được cách thức hoạt động của điểm số, với trong và ngoài mẫu huấn luyện.

![](/assets/images/fig 7-6.svg)
<details class="fig"><summary>Hình 7.6: Deviance của trong và ngoài mẫu. Ở mỗi biểu đồ, mô hình với số lượng predictor khác nhau được thể hiện ở trục hoành. Deviance qua 10000 mẫu mô phỏng nằm ở trục tung. Màu xanh là deviance trong mẫu, mầu đen là deviance ngoài mẫu. Điểm giữa là trung bình, và đường thẳng cho biết khoảng $\pm1$ độ lệch chuẩn.</summary>
<pre><code>def model(x_train, y_train, b_sigma):
    a = numpyro.param("a", jnp.array([0.0]))
    Bvec = a
    k = x_train.shape[1]
    if k > 1:
        b = numpyro.sample("b", dist.Normal(0, b_sigma).expand([k - 1]))
        Bvec = jnp.concatenate([Bvec, b])
    mu = jnp.matmul(x_train, Bvec)
    numpyro.sample("y", dist.Normal(mu, 1), obs=y_train)


def sim_train_test(N, k, i, rng_key, rho=[0.15, -0.4]):
    ## k is number of params
    n_dim = max(k, 3)
    ## covariance matrix corresponding to n_dim
    Rho = jnp.identity(n_dim)
    Rho = ops.index_update(Rho, ops.index[1 : len(rho) + 1, 0], jnp.array(rho))
    Rho = ops.index_update(Rho, ops.index[0, 1 : len(rho) + 1], jnp.array(rho))
    ## sampling 20 with covariance matrix
    mm = dist.MultivariateNormal(jnp.zeros(n_dim), Rho).sample(
        random.fold_in(random.PRNGKey(rng_key), i), (N,)
    )
    x = jnp.ones((N, 1)) # constant for intercept included
    if k > 1:
        x = jnp.concatenate([x, mm[:, 1:k]], axis=1)
    y = mm[:,0]
    return x, y


def fit(model, x_train, y_train, b_sigma, i, rng_key, scalar=True):
    k = x_train.shape[1]
    rng = random.fold_in(random.PRNGKey(rng_key), i)
    ## create a guide
    if k > 1:
        guide = AutoLaplaceApproximation(
            model, init_loc_fn=init_to_value(values={"b": jnp.zeros(k - 1)})
        )
    else:
        guide = lambda x_train, y_train, b_sigma: None
    svi = SVI(
        model, guide, optim.Adam(0.3), Trace_ELBO(),
        x_train=x_train,
        y_train=y_train,
        b_sigma=b_sigma
    )
    init_state = svi.init(rng)
    state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
    params = svi.get_params(state)
    if scalar:
        coefs = params['a']
        if k > 1:
            coefs = jnp.concatenate([coefs, guide.median(params)["b"]])
        return coefs
    else:
        nums = 1000
        a_posterior = jnp.repeat(params['a'],nums).reshape(-1,1)
        if k > 1:
            b_posterior = guide.sample_posterior(rng, params, (nums,))['b']
        else:
            b_posterior = None
        return {"a":a_posterior, "b":b_posterior}



def true_deviance_train_test(N, k, b_sigma, i):
    # create sample
    x_train, y_train = sim_train_test(N, k, i, rng_key=0)
    x_test, y_test = sim_train_test(N, k, i, rng_key=1)
    # fit
    coefs = fit(model, x_train, y_train, b_sigma, i, rng_key=2)
    # deviance train
    mu = jnp.matmul(x_train, coefs)
    logprob = dist.Normal(mu).log_prob(y_train)
    dev_train = (-2) * jnp.sum(logprob)
    # deviance test
    mu = jnp.matmul(x_test, coefs)
    logprob = dist.Normal(mu).log_prob(y_test)
    dev_test = (-2) * jnp.sum(logprob)
    return jnp.stack([dev_train, dev_test])

def dev_fn(N, k, b_sigma):
    r = vmap(lambda i: true_deviance_train_test(N, k, b_sigma, i))(jnp.arange((int(1e4))))
    # return  dev: train_mean, test_mean, train_std, test_std
    return jnp.concatenate([jnp.mean(r, 0), jnp.std(r, 0)])

fig, axes = plt.subplots(1,2, figsize=(15,7))
kseq = range(1, 6)

for ax, N in zip(axes, [20, 100]):
    dev = jnp.stack([dev_fn(N, k, 100) for k in kseq], axis=1)
    ax.set(
        ylim=(jnp.min(dev[0]).item() - 5, jnp.max(dev[0]).item() + 12),
        xlim=(0.9, 5.2),
        xlabel="number of parameters",
        ylabel="deviance",
    )
    ax.set_title("N = {}".format(N))
    ax.scatter(jnp.arange(1, 6), dev[0], s=80, color="b")
    ax.scatter(jnp.arange(1.1, 6), dev[1], s=80, color="k")
    pts_int = (dev[0] - dev[2], dev[0] + dev[2])
    pts_out = (dev[1] - dev[3], dev[1] + dev[3])
    ax.vlines(jnp.arange(1, 6), pts_int[0], pts_int[1], color="b")
    ax.vlines(jnp.arange(1.1, 6), pts_out[0], pts_out[1], color="k")
    ax.annotate(
        "in", (2, dev[0][1]), xytext=(-25, -5), textcoords="offset pixels", color="b"
    )
    ax.annotate("out", (2.1, dev[1][1]), xytext=(10, -5), textcoords="offset pixels")
    ax.annotate(
        "+1SD",
        (2.1, pts_out[1][1]),
        xytext=(10, -5),
        textcoords="offset pixels",
        fontsize=12,
    )
    ax.annotate(
        "-1SD",
        (2.1, pts_out[0][1]),
        xytext=(10, -5),
        textcoords="offset pixels",
        fontsize=12,
    )
plt.tight_layout()</code></pre></details>

Hình trên cho kết quả 10000 mẫu mô phỏng với mỗi loại mô hình, ở hai cỡ mẫu khác nhau. Hàm số thực hiện mô phỏng là `sim_train_test`. Nếu bạn muốn thực hiện mô phỏng như dạng này, hãy xem phần code trên. Trục tung được thể hiện ở thang đo $-2 \times \text{lppd}$, "deviance", nên số càng lớn thì càng tệ. Ở biểu đồ bên trái, cả mẫu huấn luyện và mẫu kiểm tra đều chứa 20 trường hợp. Màu xanh và đường thẳng thể hiện trung bình cộng-và-trừ một đơn vị độ lệch chuẩn của deviance tính được trên mẫu huấn luyện. Từ trái qua phải thì số lượng parameter tăng lên, deviance trung bình thì giảm. Deviance nhỏ hơn nghĩa là fit tốt hơn. Vậy sự giảm xuống theo độ phức tạp mô hình, là hiện tượng tương tự bạn đã thấy trước đó vởi $R^2$.

Nhưng nếu bạn soi kỹ vào những điểm và đường thẳng đen. Biểu đồ này cho thấy phân phối deviance ngoài mẫu ứng với số lượng parameter. Trong khi deviance trong mẫu luôn tốt hơn khi số lượng parameter thêm vào, deviance ngoài mẫu có trung bình nhỏ nhất với 3 parameter, đúng với mô hình tạo data trong ví dụ này. Deviance ngoài mẫu tệ hơn (tăng lên) khi số lượng parameter thêm vào sau thứ ba. Những parameter thêm vào này fit những yếu tố gây nhiễu (noise) do predictor thêm vào. Cho nên khi deviance tiếp tục cải thiện (giảm xuống) ở mẫu huấn luyện, nó tệ hơn ở mẫu kiểm tra. Biểu đồ bên phải cũng cho hiện tượng tương tự với cỡ mẫu lớn hơn $N=100$ trường hợp.

Kích thước của cột độ lệch chuẩn có thể làm bạn ngạc nhiên. Trong khi nó luôn luôn đúng khi deviance ngoài mẫu tệ hơn deviance trong mẫu, mỗi cặp cụ thể của mẫu huấn luyện và kiểm tra cí thể đảo ngược sự mong đợi này. Lý do là với một mẫu huấn luyện bất kỳ đều có thể gây hiểu nhầm. Và một mẫu kiểm tra bất kỳ đều có thể không mang tính đại diện. Hãy giữ sự thật này trong đầu khi chúng ta phát triển công cụ để so sánh mô hình, bởi vì sự thật này sẽ cản trở bạn thêm quá nhiều niềm tin vào phân tích của bất kỳ mẫu nào. Giống như tất cả suy luận thống kê, không có sự đảm bảo nào ở đây.

Cũng ghi chú thêm, không có gì đảm bảo rằng mô hình tạo data "thực" sẽ có deviance ngoài mẫu nhỏ nhất. Bạn có thể thấy triệu chứng của sự thật này ở deviance của mô hình có 2 parameter. Mô hình có dự đoán tệ hơn so với mô hình có 1 parameter, mặc dù mô hình thực sự bao gồm cả biến dự đoán thêm vào. Đó là bởi vì với $N=20 $ trường hợp, độ chính xác của ước lượng cho predictor đầu tiên tạo ra nhiều sai lệch hơn so với khi bỏ mặc nó. Ở biểu đồ bên phải, ngược lại, nó có đủ data để ước lượng chính xác quan hệ giữa biến predictor thứ nhất và biến kết cục. Bây giờ deviance cho mô hình 2 parameter tốt hơn so với mô hình 1 parameter.

Deviance là một đại lượng đánh giá độ chính xác của dự đoán, không phải đánh giá sự thật. Mô hình thực, tức là mô hình chứa những predictor đúng, không được đảm bảo là tạo ra độ chính xác tốt nhất. Tương tự với mô hình sai, không chắc chắn sẽ tạo dự đoán kém.

Điểm chính của bài mô phỏng này là diễn giải cơ chế hoạt động của deviance, theo lý thuyết. Trong khi deviance ở data huấn luyện luôn cải thiện với số lượng biến dự đoán nhiều hơn, deviance ở data tương lai có thể không như vậy, phụ thuộc vào mô hình xử lý thực sự tạo data và bao nhiêu data là có sẵn để ước lượng chính xác parameter. Sự thật này là nền tảng để hiểu rõ regularing prior và tiêu chuẩn thông tin.

## <center>7.3 Huấn luyện golem: Regularizing</center><a name="a3"></a>

Nếu tôi nói rằng một cách để cho dự đoán tốt hơn là làm cho mô hình fit data kém hơn? Bạn sẽ tin nó không? Trong phần này, tôi sẽ trình diễn nó.

Nguồn gốc của overfitting là mô hình luôn có tình trạng quá phấn khởi bởi mẫu huấn luyện. Khi prior là phẳng hay gần phẳng, bộ máy diễn giải điều này có nghĩa là mọi giá trị parameter đều có khả năng xảy ra như nhau. Kết quả là, mô hình sẽ cho posterior mã hoá hầu hết mẫu huấn luyện càng nhiều càng tốt - như đã được thể hiện qua hàm likelihood.

Một cách để chống lại việc mô hình bị quá phấn khởi bởi mẫu huấn luyện là sử dụng prior đa nghi. "Đa nghi" ở đây là prior có thể làm chậm tốc độ học từ mẫu. Prior đa nghi thông dụng nhất là **REGULARIZING PRIOR**. Prior này, khi được tinh chỉnh tốt, sẽ giảm thiểu overfitting trong khi cho phép mô hình học những *đặc trưng thường* của mẫu. Nếu prior quá đa nghi, tuy nhiên, thì đặc trưng thường có thể bị bỏ quả, gây ra underfitting. Vậy vấn đề thực sự ở đây là tinh chỉnh. Những như bạn sẽ thấy, ngay cả sự đa nghi cũng có thể giúp mô hình tốt hơn, và làm tốt hơn là tất cả những gì chúng ta hi vọng ở thế giới lơn, nơi mà không một mô hình hay prior nào là tối ưu nhất.

Ở những chương trước, tôi bắt các bạn tuỳ chỉnh prior cho đến khi phân phối dự đoán prior tạo ra được những kết cục có lý. Hệ quả là, những prior đó làm thông dụng hoá (regularize) suy luận. Ở cỡ mẫu nhỏ, chúng là sự giúp đõ to lớn. Ở đây tôi muốn cho các bạn thấy lý do, bằng cách sử dụng nhiều tình huống mô phỏng. Xem xét mô hình Gaussian này:

$$ \begin{aligned}
y_i &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta x_i \\
\alpha &\sim \text{Normal}(0 ,100)\\
\beta &\sim \text{Normal} (0,1)\\
\sigma &\sim \text{Exponential} (1) \\
\end{aligned}$$

Giả định, như thói quen tốt, rằng biến dự đoán $x$ được chuẩn hoá để độ lệch chuẩn là 1 và trung bình là zero. Thì prior của $\alpha$ gần như là prior phẳng và không có hiệu ứng thực tế nào trong suy luận, như bạn đã xem ở những chương trước.

Nhưng prior của $\beta$ thì hẹp hơn và được dùng để regularize. Prior $\beta \sim \text{Normal} (0,1)$ nói rằng, trước khi thấy data, cỗ máy sẽ rất hoài nghi những giá trị trên 2 và dưới -2, vì prior Gaussian có độ lệch chuẩn là 1 chỉ gán 5% xác suất cho giá trị trên và dưới 2 đơn vị độ lệch chuẩn. Bởi vì biến dự đoán $x$ được chuẩn hoá rồi, bạn có thể diễn giải rằng là sự thay đổi 1 độ lệch chuẩn ở $x$ thì rất hiếm tạo ra 2 đơn vị thay đổi ở kết cục.

![](/assets/images/fig 7-7.svg)
<details class="fig"><summary>Hình 7.7: Regularizing prior, yếu và mạnh. Ba prior có độ lệch chuẩn khác nhau. Những prior này sẽ giúp giảm overfitting, nhưng ở mức độ khác nhau. Đường nứt: Normal(0, 1). Đường liền mỏng: Normal(0, 0.5). Đường liền dày: Normal(0, 0.2).</summary>
<pre><code>ax = plt.gca()
x = jnp.linspace(-3,3,100)
y1 = jnp.exp(dist.Normal(0, 1).log_prob(x))
y2 = jnp.exp(dist.Normal(0, 0.5).log_prob(x))
y3 = jnp.exp(dist.Normal(0, 0.2).log_prob(x))
ax.plot(x, y1, linestyle='dashed')
ax.plot(x, y2, linewidth=2)
ax.plot(x, y3, linewidth=1)</code></pre></details>

Bạn có thể hiện prior này trên biểu đồ như đường nứt ở hình trên. Khi càng nhều xác suất tụ lại quanh zero, ước lượng thì thu nhỏ lại gần zero - nó mang tính bảo tồn. Những đường cong khác là những prior hẹp hơn và đa nghi hơn những giá trị parameter xa zero. Đường liền mỏng là prior Gaussian mạnh hơn với độ lệch chuẩn 0.5. Đường liền dày càng mạnh hơn, với độ lệch chuẩn chỉ 0.2.

Mức độ mạnh hay yếu của prior đa nghi này ở thực tế sẽ phụ thuộc vào data và mô hình. Hãy khám phá ví dụ sau đây, nó khá giống như ví dụ phần trước. Lần này chúng ta sẽ dùng regularizing prior như hình 7.7, thay vì những prior phẳng. Cho mỗi một trong năm mô hình khác nhau, ta mô phỏng 10000 lần cho mỗi regularizing prior trên. Hình 7.8 là kết quả của mô phỏng. Những điểm trong hình là deviance của prior phẳng như phần trước: màu xanh cho deviance huấn luyện và màu đen cho deviance kiểm tra. Đường thẳng biểu diễn cho deviance huấn luyện và kiểm tra với prior khác nhau. Đường màu xanh là deviance huấn luyện và đường màu đen là deviance kiểm tra. Hình dạng của những đường thẳng tương ứng như hình 7.7.

![](/assets/images/fig 7-8.svg)
<details class="fig"><summary>Hình 7.8: Regularizing prior và deviance ngoài mẫu. những điểm trong hai hình là giống như hình 7.6. Đường thẳng </summary>
<pre><code>fig, axes = plt.subplots(1,2, figsize=(10,7))
kseq = range(1, 6)

for ax, N in zip(axes, [20, 100]):
    dev1 = jnp.stack([dev_fn(N, k, 1) for k in kseq], axis=1)
    dev2 = jnp.stack([dev_fn(N, k, 0.5) for k in kseq], axis=1)
    dev3 = jnp.stack([dev_fn(N, k, 0.2) for k in kseq], axis=1)
    ax.set(
        ylim=(jnp.min(dev1[0]).item() - 5, jnp.max(dev1[0]).item() + 12),
        xlim=(0.9, 5.2),
        xlabel="number of parameters",
        ylabel="deviance",
    )
    ax.set_title("N = {}".format(N))
    ax.plot(jnp.arange(1, 6), dev1[0], color="b", linestyle="dashed")
    ax.plot(jnp.arange(1.1, 6), dev1[1], color="k", linestyle="dashed", label="Normal(0 ,1)")
    ax.plot(jnp.arange(1, 6), dev2[0], color="b", linewidth=1)
    ax.plot(jnp.arange(1.1, 6), dev2[1], color="k", linewidth=1, label="Normal(0 ,0.5)")
    ax.plot(jnp.arange(1, 6), dev3[0], color="b", linewidth=2)
    ax.plot(jnp.arange(1.1, 6), dev3[1], color="k", linewidth=2, label="Normal(0 ,0.2)")
    ax.legend()
plt.tight_layout()</code></pre></details>

Nhìn vào biểu đồ bên trái, với cỡ mẫu $N=20$. Deviance huấn luyện luôn luôn tăng - tệ hơn - với prior hẹp hơn. Đường xanh dày là lớn hơn hẳn những đường khác, và bởi vì prior đa nghi ngăn cản mô hình học toàn bộ mẫu. Nhưng với deviance kiểm tra, ngoài mẫu, thì nó cải thiện (nhỏ hơn) với prior hẹp hơn. Mô hình có ba parameter vẫn là mô hình ngoài mẫu tốt nhất, và regularizing prior có một ít tác động lên deviance của nó.

Nhưng cũng để ý rằng khi prior càng đa nghi hơn, tổn thương gây ra bởi mô hình phức tạp quá sẽ được giảm đi rất nhiều. Với prior Normal(0, 0.2) (đường dày), mô hình với 4 và 5 parameter chỉ tệ hơn mô hình đúng một chút. Nếu bạn có thể tinh chỉnh đúng regularizing prior, thì overfitting sẽ được giảm thiểu đáng kể.

Giờ tập trung vào biểu đồ bên phải, khi cỡ mẫu là $N=100$. Prior giờ đây có ít ảnh hưởng hơn, bởi vì chúng ta có nhiều bằng chứng hơn. Những prior này vẫn có ích. Nhưng overfitting giờ ít nghiêm trọng hơn, và có đủ thông tin ở data để vượt qua ngay cả prior Normal(0, 0.2).

Regularizing prior là tốt, bởi vì chúng giúp giảm overfitting. Nhưng nếu chúng quá đa nghi, chúng ngăn cản mô hình học data. Khi bạn gặp mô hình đa tầng ở Chương 13, bạn sẽ thấy cỗ máy trung tâm của chúng là tự học độ mạnh của prior từ chính data. Cho nên bạn có thể suy nghĩ rằng mô hình data là sự regularizing tự thích nghi (adaptive regularizing), khi mô hình tự nó cố gắng điều chỉnh sự đa nghi.

<div class="alert alert-info">
    <p><strong>Ridge regression.</strong> Mô hình tuyến tính với parameter của slope dùng prior Gaussian, điểm giữa ở zero, đôi khi được biết đến là <strong>HỒI QUY RIDGE (RIDGE REGRESSION)</strong>. Ridge regression thông thường sẽ nhận thêm giá trị độ chính xác $\lambda$, nó mô tả độ hẹp của các prior. $\lambda>0$ sẽ giúp giảm overfitting. Tuy nhiên, cũng giống trong phiên bản Bayes, nếu $\lambda$ quá lớn, chúng ta có nguy cơ bị underfitting. Trong khi không bắt nguồn từ Bayes, ridge regression là một ví dụ khác về hiện tượng quy trình thống kê có thể được hiểu theo góc nhìn Bayes và non-Bayes. Ridge regression không có tính phân phối posterior. Thay vào đó, nó dùng một phiên bản khác của OLS có thêm $\lambda$ vào trong phép tính đại số ma trận để ra ước lượng.</p>
    <p> Mặc dù rất dễ để dùng regularization, đa só phương pháp thống kê cổ điển không dùng nó. Nhà thống kê thường trêu machine learning chỉ là tái chế lại thống kê với tên mới. Nhưng regularization là một lĩnh vực trong machine learning mà phát triển hơn. Khoá học machine learning cơ bản thường mô tả regularization. Nhưng thống kê cơ bản thì không.</p>
</div>

## <center>7.4 Dự đoán độ chính xác của dự đoán</center><a name="a4"></a>

Tất cả những phần trên cho thấy có một đường đi giữa overfitting và underfitting: Lượng giá mô hình ngoài mẫu. Những chúng ta không có những data ngoài mẫu, về định nghĩa, làm sao chúng ta lượng giá mô hình trên nó? Có hai nhánh phương pháp thực hiện điều này: **CROSS-VALIDATION** và **INFORMATION CRITERIA**. Những phương pháp này thử đoán mô hình sẽ hoạt động tốt như thế nào, theo trung bình, trong dự đoán data mới. Chúng ta sẽ xem xét cả hai tiếp cận chi tiết hơn. Mặc dù có sự khác nhau cơ bản trong các phép tính toán học của chúng, chúng cho ra những ước lượng rất gần nhau.

### 7.4.1 Cross-validation

Một phương pháp thông dụng để ước lượng độ chính xác dự đoán là thực sự kiểm tra độ chính xác dự đoán trên một mẫu khác. Còn gọi là **CROSS-VALIDATION**, nó lấy ra một bộ phận nhỏ của mẫu quan sát và lượng giá mô hình dựa trên mẫu bị lấy ra đó. Dĩ nhiên là chúng ta không muốn từ chối data. Vậy cái gì thường được dùng để trách mẫu ra thành từng data nhỏ, gọi là "fold". Mô hình sẽ được yêu cầu dự đoán các fold, sau khi được huấn luyện trên phần còn lại. Chúng ta sẽ lấy trung bình các điểm số của từng fold và tính ra độ chính xác dự đoán mong đợi. Số lượng fold tối thiểu là 2. Ở cực trị, bạn có thể lấy một mẫu quan sát làm một fold và fit mô hình với số lượng mô hình bằng cỡ mẫu.

Bạn nên sử dụng bao nhiêu fold? Đây là câu hỏi ít được nghiên cúu. Rất nhiều khuyến cáo nói rằng nhiều quá hoặc ít quá sẽ cho ước lượng độ chính xác dự đoán ngoài mẫu ít tin cậy hơn. Nhưng những nghiên cứu mô phỏng không cho rằng như vậy.<sup><a name="r113" href="#113">113</a></sup> Việc dùng số lượng fold lớn nhất là cực kỳ phổ biến, nghĩa là tách một mẫu độc nhất thành từng fold. Nó còn gọi là **LEAVE-ONE-OUT CROSS-VALIDATION** (thường được gọi tắt là LOOCV). LOOCV là thứ chúng ta sẽ học ở chương này.

Câu hỏi chính của LOOCV là, nếu chúng ta có 1000 quan sát, có nghĩa ta phải tính phân phối posterior 1000 lần. Điều này rất mất thời gian. May mắn thay, có nhiều cách rất hay để ước lượng điểm LOOCV mà không cần phải chạy mô hình nhiều lần như vậy. Một cách tiếp cận là dùng "độ quan trọng (importance)" cho mỗi quan sát cho phân phối posterior. "Độ quan trọng" ở đây nghĩa là có vài quan sát có mức độ ảnh hưởng lớn hơn trong phân phối posterior - nếu chúng ta loại một quan sát quan trọng, posterior sẽ thay đổi nhiều. Những quan sát khác sẽ ít ảnh hưởng hơn. Đây là một khía cạnh lành tính của vũ trụ khi mà độ quan trọng có thể được ước lượng mà không cần fit lại mô hình.<sup><a name="r114" href="#114">114</a></sup> Có thể hiểu đơn giản là có một quan sát ít xảy ra nhưng quan trọng hơn những quan sát khác mà được mong đợi dễ xuất hiện hơn. Khi không được như sự mong đợi, bạn phải thay đổi sự mong đợi nhiều hơn. Suy luận Bayes hoạt động tương tự. Độ quan trọng thường được gọi là *trọng số (weight)*, và những trọng số này có thể dùng để ước lượng độ chính xác ngoài mẫu của mô hình.

Với hàng tá chi tiết toán học được giấu dưới tấm thảm, phương pháp này cho kết quả ước lượng điểm cross-validation rất tốt. Nó có một cái tên khá bất tiện là **PARETO-SMOOTHED IMPORTANCE SAMPLING CROSS-VALIDATION**.<sup><a name="r115" href="#115">115</a></sup>  Chúng ta sẽ gọi là **PSIS** cho gọn. PSIS sử dụng lấy mẫu độ quan trọng (importance sampling), hay có nghĩa là sử dụng trọng số độ quan trọng được mô tả ở trên. Kỹ thuật làm mượt Pareto (Pareto-smoothing) là kỹ thuật làm cho trọng số độ quan trọng được đáng tin cậy hơn. Pareto là tên của một thành phố nhỏ ở Bắc Ý. Nhưng nó cũng là tên của nhà khoa học người Ý, Vilfredo Pareto (1848-1923), người đã đóng góp rất nhiều công trình khoa học quan trọng. Một trong những công trình đó là **PHÂN PHỐI PARETO**. PSIS sử dụng phân phối này để suy ra điểm số cross-validation đáng tin cậy hơn, mà không cần phải thực hiện cross-validation trực tiếp. Nếu bạn muốn biết chi tiết hơn, mời xem phần dưới.

Tính năng quan trọng nhất của PSIS là nó cho phép phản hồi về sự tin cậy của nó. Nó thực hiện điều này bằng cách ghi chú lại quan sát mà có trọng số rất cao, làm cho điểm số PSIS không chính xác. Chúng ta sẽ xem phần thực hành ở dưới để biết chi tiết hơn.

Một tính năng khác của cross-validation và ước lượng PSIS là nó thực hiện phép tính theo từng mẫu quan sát (pointwise). Tính chất từng điểm cho phép ước lượng chính xác - đôi khi rất chính xác - sai số chuẩn (standard error) của ước lượng deviance ngoài mẫu. Để tính sai số chuẩn này, chúng ta tính điểm CV hoặc PSIS cho mỗi mẫu quan sát và lợi dụng central limit theorem để tính:

$$ s_{\Tiny PSIS} = \sqrt{N \text{var} (\text{psis}_i) } $$

Trong đó, $N$ là cỡ mẫu và $\text{psis}_i$ là ước lượng PSIS cho quan sát thứ $i$. Nếu bạn vẫn chưa hiểu, hãy xem phần code ở cuối phần này.

<div class="alert alert-dark">
	<p><strong>Pareto-smoothed cross-validation</strong>. Cross-validation ước lượng lppd của data ngoài mẫu. Nếu bạn có $N$ mẫu quan sát và fit mô hình $N$ lần, mỗi lần bỏ ra một quan sát $y_i$, thì lppd ngoài mẫu là tổng của độ chính xác cho mỗi $y_i$ bỏ ra.
	$$ \text{lppd}_{\Tiny CV} = \displaystyle\sum_{i=1}^N \frac{1}{S} \displaystyle\sum_{s=1}^S \log \Pr(y_i | \theta_{-i,s}) $$
	trong đó $s$ là số thứ tự mẫu từ chuỗi Markov và $\theta_{-i,s}$ là mẫu thứ $s$ từ phân phối posterior tính từ mẫu quan sát không chứa $y_i$.</p>
	<p>Lấy mẫu độ quan trọng thay thế cho những phép tính toán $N$ lần phân phối posterior bằng cách sử dụng ước lượng của độ quan trọng của mỗi $i$ vào phân phối posterior. Chúng ta lấy mẫu từ toàn bộ phân phối posterior $p(\theta|y)$, nhưng chúng ta muốn mẫu từ phân phối posterior đã được lấy một mẫu ra (leave-one-out) $p(\theta|y_{-i})$. Cho nên chúng ta thiết lập lại trọng số cho mỗi mẫu $s$ bằng đảo ngược xác suất của mẫu quan sát bị loại bỏ ra:<sup><a name="r116" href="#116">116</a></sup>
	$$ r(\theta_s) = \frac{1} {p(y_i|\theta_s)} $$
	Trọng số này mang tính tương đối, nhưng nó được bình thường hóa (normalized) trong công thức dưới đây:
	$$ \text{lppd}_{\Tiny IS} = \displaystyle\sum_{i=1}^N \log \frac{\sum_{s=1}^S r(\theta_s)p(y_i|\theta_s)}{\sum_{i=1}^S r(\theta_s)} $$
	Và đây là lppd ngoài mẫu ước lượng từ lấy mẫu độ quan trọng (importance sampling).</p>
	<p>Chúng tôi vẫn chưa làm gì bằng kỹ thuật làm mượt Pareto. Lý do chúng cần phải sử dụng nó là những trọng số $r(\theta_s)$ có thể không đáng tin cậy. Cụ thể, nếu bất kỳ $r(\theta_s)$ nào quá lớn, nó sẽ làm hại lppd bằng cách chiếm dụng. Một phương pháp để giới hạn lại các trọng số là không trọng số nào theo lý thuyết vượt qua một ngưỡng giới hạn. Nó có ích, nhưng nó cũng ảnh hưởng sai lệch đến ước lượng. PSIS thì thông minh hơn. Nó khai thác một sự thật rằng phân phối của các trọng số tuân theo một hình dạng nhất định, dưới một số điều kiện thường gặp. Trọng số lớn nhất sẽ theo luật <strong>PHÂN PHỐI PARETO</strong>:
	$$ p(r| \mu, \sigma, k) = \sigma^{-1} (1 + k(r-u)\sigma^{-1})^{-\frac{1}{k} -1} $$
	trong đó $\mu$ là parameter vị trí (location), $\sigma$ là độ lệch chuẩn (hay scale), $k$ là hình dạng (shape). Với mỗi quan sát $y_i$, trọng số lớn nhất sẽ được dùng để ước lượng phân phối Pareto và sau đó được làm mượt bằng phân phối Pareto đó. Theo lý thuyết và trong thực hành, kỹ thuật này đều hoạt động tốt.<sup><a name="r117" href="#117">117</a></sup> Điều hay nhất ở cách tiếp cận này là ước lượng cho $k$ cho thông tin về khoảng tin cậy của dự đoán. Nó sẽ có một giá trị $k$ cho mỗi giá trị $y_i$. Giá trị $k$ lớn hơn chỉ điểm những quan sát có mức độ ảnh hưởng lớn, và nếu $k>0.5$, thì phân phối Pareto có phương sai vô hạn. Phân phối có phương sai vô hạn thì có đuôi rất dày. Bởi vì chúng ta cố gắng làm mượt trọng số độ quan trọng bằng đuôi của phân phối, phương sai vô hạn làm cho trọng số khó tin cậy hơn. Cho nên, cả theo lý thuyết và mô phỏng, trọng số PSIS hoạt động tốt hơn chỉ khi $k<0.7$. Khi chúng ta bắt đầu dùng PSIS, bạn sẽ thấy cảnh báo những giá trị $k$ lớn. Nó có ích cho việc phát hiện những quan sát có ảnh hướng lớn.</p>
</div>

### 7.4.2 Tiêu chuẩn thông tin (Information criteria)

Cách tiếp cận thứ hai là sử dụng **TIÊU CHUẨN THÔNG TIN (INFORMATION CRITERIA)** để tính điểm số mong đợi ngoài mẫu. Tiêu chuẩn thông tin thiết lập một giá trị ước lượng độ phân kỳ KL ngoài mẫu tương đối theo lý thuyết.

Nếu bạn nhìn lại hình 7.8, có một quy luật ở khoảng cách giữa các điểm (cặp huấn luyện-kiểm tra với prior phẳng): Hiệu của chúng gần bằng hai lần số lượng parameter tương ứng với mỗi mô hình.  Hiệu giữa deviance huấn luyện và deviance kiểm tra là gần bằng 2 ở mô hình đầu tiên (1 parameter) và khoảng 10 cho mô hình cuối (5 parameter). Điều này không phải là sự trùng hợp, nó là một hiện tượng độc đáo trong machine learning: Với hồi quy tuyến tính thông thường với prior phẳng, giá trị phạt đền cho overfitting thì gần bằng hai lần số lượng parameter.

Đây là một hiện tượng đứng sau **TIÊU CHUẨN THÔNG TIN**. Tiêu chuẩn được biết nhiều nhất là **TIÊU CHUẨN THÔNG TIN AKAIKE (AKAIKE INFORMATION CRITERION)**, viết tắt là **AIC**.<sup><a name="r118" href="#118">118</a></sup> AIC cung cấp kết quả ước lượng cho deviance ngoài mẫu một cách khá ngạc nhiên:

$$ \text{AIC} = D_{\text{train}} + 2p = -2 \text{lppd} + 2p $$

Trong đó, $p$ là số lượng parameter tự do ở posterior. Con số 2 ở đó là dành cho chuẩn hóa lại, những gì AIC nói cho chúng ta biết là số chiều không gian của posterior là thước đo tự nhiên cho overfitting của mô hình. Những mô hình phức tạp hơn thường dễ bị overfitting, tỉ lệ trực tiếp với số lượng parameter.

AIC giờ đây đã là lịch sử. Những phương pháp ước lượng tân tiến và tổng quát hơn đang tồn tại và trội hơn AIC ở mọi mặt. Nhưng Akaike nên được tuyên dương bởi những cảm hứng ban đầu xuất phát từ ông. Hãy xem phần dưới để biết thêm chi tiết. AIC là con số ước lượng đáng tin cậy khi và chỉ khi:
1. Prior phẳng hoặc bị likelihood vượt trội nhiều.
2. Phân phối posterior gần bằng Gaussian đa biến.
3. Cỡ mẫu $N$ lớn hơn rất nhiều<sup><a name="r119" href="#119">119</a></sup> so với số lượng biến.

Bởi vì prior phẳng không bao giờ là prior tốt nhất, chúng ta muốn một thứ gì khác tổng quát hơn. Khi bạn học tới mô hình đa tầng, prior không bao giờ phẳng về mặt định nghĩa. Có một tiêu chuẩn tổng quát hơn là **TIÊU CHUẨN DEVIANCE (DEVIANCE INFORMATION CRITERION - DIC)**. DIC dùng được với những prior chứa thông tin, nhưng vẫn giả định rằng posterior là Gaussian đa biến và $N \gg k$.<sup><a name="r120" href="#120">120</a></sup>

<div class="alert alert-dark">
	<p><strong>Cảm hứng của Akaike.</strong> AIC là một phát minh tuyệt vời. Hitotugu Akaike (赤池弘次, 1927–2009) đã giải thích ý tưởng xuất phát từ đâu: "Vào một buổi sáng ngày 16-03-1971, khi đang ngồi trên xe lửa đi làm, tôi chợt nhận ra rằng parameter trong mô hình phân tích yếu tố được ước lượng bằng tối đa hóa likelihood và giá trị trung bình của logarith của likelihood liên kết với số lượng thông tin Kullback-Leiber."<sup><a name="r121" href="#121">121</a></sup> Con tàu ấy chắc rất đặc biệt. Cái gì trong đầu ông khi ông nhận ra điều đó? Về mặt kỹ thuật, việc suy ra AIC cần phải viết mục tiêu trước, tức là độ phân kỳ KL mong đợi, và sau đó thực hiện ước lượng. Kết quả sai lệch mong đợi sẽ tỉ lệ với số lượng parameter, cho rằng dưới một số giải định cần thiết.</p>
</div>

Chúng ta sẽ tập trung vào một tiêu chuẩn tổng quát hơn cả AIC và DIC. **Widely Applicable Information Criterion (WAIC)** của Sumio Watanabe (渡辺澄夫) không cần phải giả định gì về hình dạng của posterior.<sup><a name="r122" href="#122">122</a></sup> Nó trả kết quả ước lượng deviance ngoài mẫu mà tương đương với ước lượng cross-validation ở cỡ mẫu lớn. Nhưng ở cỡ mẫu nhất định, nó có thể sẽ không đồng thuận. Nó không đồng thuận bởi vì nó có mục tiêu khác - nó không cố gắng ước lượng điểm cross-validation, mà nó muốn ước lượng độ phân kỳ KL ngoài mẫu. Ở cỡ mẫu lớn, chúng thường bằng nhau.

Chúng ta tính WAIC như thế nào? Thật không may, tính năng tổng quát của nó phải đánh đổi bằng công thức phức tạp hơn. Nhưng nó thực chất gồm hai thành phần, và bạn có thể tính chúng từ mẫu của posterior. WAIC chỉ là lppd mà chúng ta sử dụng ở phần trước, cộng thêm một giá trị phạt đền tỉ lệ với phương sai của dự đoán posterior:

$$ \text{WAIC} (y, \Theta) = -2\big( \text{lppd} - \underbrace{\displaystyle\sum_i \text{var}_\theta \log p(y_i | \theta}_\text{số hạng phạt đền} ) \big) $$

Trong đó, $y$ là mẫu quan sát và $\Theta$ là phân phối posterior. Giá trị phạt đền có nghãi là, "tính ra phương sai của logarith xác suất cho mỗi quan sát $i$, và sau đó cộng chúng lại để có được giá trị phạt đền tổng cộng." Bạn có thể nghĩ mỗi quan sát tự có giá trị phạt đền riêng. Và vì điểm số này đo lường nguy cơ overfitting, bạn có thể tiếp cận nguy cơ overfitting ở mức độ từng quan sát.

Bởi vì tính chất tương tự với tiêu chuẩn nguyên bản của Akaike, số hạng phạt đền trong WAIC đôi khi được gọi **SỐ LƯỢNG PARAMETER HIỆU QUẢ (EFFECTIVE NUMBER OF PARAMETERS)**, ký hiệu $p_{WAIC}$. Ký hiệu này mang tính lịch sử, nhưng không có ý nghĩa gì về toán học. Như chúng ta sẽ thấy, nguy cơ overfitting của mô hình không liên quan nhiều đến số lượng parameter hơn là cách parameter quan hệ với nhau như thế nào. Khi chúng ta học tới mô hình đa tầng, việc thêm parameter vào mô hình thực ra sẽ *giảm thiểu* "số lượng parameter hiệu quả". Cũng giống như tiếng anh, trong thống kê có rất nhiều rác lịch sử làm cản trở việc học hành. Không ai muốn chuyện đó cả. Nó chỉ là sự tiến bộ văn hóa. Chúng ta sẽ cố gắng gọi số hạng phạt đền là "phạt đền cho overfitting." Nhưng nếu bạn thấy nó dưới tên gọi số lượng parameter hiệu quả ở nơi khác, bạn nên biết chúng là một.

Nếu bạn muốn biết rõ cách tính lppd và số hạng phạt đền, bạn có thể xem thêm ở phần dưới. Nhìn công thức toán học ở trên thông qua mã máy tính có thể giúp bạn hiểu thông hơn.

Giống như PSIS, WAIC là *theo từng điểm (pointwise)*. Dự đoán được xem xét theo từng trường hợp, hay từng mẫu quan sát, trong data. Nhiều thứ suất phát từ đây. Đầu tiên, WAIC sẽ cho sai số chuẩn tương đối. Thứ hai, bởi vì có vài quan sát có mức độ ảnh hưởng lớn hơn cho posterior, WAIC sẽ ghi nhận nó lại theo từng điểm trong số hạng trừng phạt của nó. Thứ ba, cũng giống như cross-validation và PSIS, bởi vì WAIC cho phép tách data thằng nhiều mẫu quan sát độc lập, đôi khi rất khó để giải thích nó. Xem sét ví dụ có một mô hình mà mỗi dự đoán phụ thuộc vào quan sát trước. Điều này xảy ra trong data kiểu *chuỗi thời gian (time series)*. Trong chuỗi thời gian, quan sát ở trước trở thành biến dự đoán cho quan sát tiếp theo. Bạn sẽ không dễ nghĩ rằng mỗi quan sát là độc lập hoặc *đổi chỗ được (exchangeable)*. Trong trường hợp đó, bạn dĩ nhiên có thể tính WAIC dưới giả định mỗi quan sát là độc lập với nhau, những kết quả sẽ không cho ý nghĩa rõ ràng.

Lời cảnh báo này đưa thêm một vấn đề chung cho mọi phương pháp dự đoán độ chính xác ngoài mẫu: Tính khả dụng của chúng phải tùy theo công việc dự đoán trong suy nghĩ của bạn. Và không phải tất cả dự đoán đều có thể sử dụng công thức mà chúng ta đang giả định cho ví dụ mô phỏng huấn luyện-kiểm tra trên. Khi chúng ta học mô hình đa tầng, vấn đề này sẽ nổi lên nữa.

<div class="alert alert-info">
	<p><strong>Tiêu chuẩn thông tin và sự kiên định.</strong> Như đã nói, tiêu chuẩn như AIC và WAIC không luôn luôn gán $D_\text{test}$ mong đợi cho mô hình "thực". Trong thống kê, tiêu chuẩn thông tin là không <strong>KIÊN ĐỊNH (CONSISTENCY)</strong> với vấn đề xác định mô hình (model identification). Những tiêu chuẩn này nhắm đến tuyển chọn mô hình có dự đoán tốt nhất, bằng việc đánh giá deviance ngoài mẫu, cho nên cũng không ngạc nhiên gì nếu chúng không hoàn thành hoặc phải làm những gì ngoài thiết kế của chúng. Những thước đo khác dành cho so sánh mô hình thì kiên định hơn. Vậy tiêu chuẩn thông tin là sai?</p>
	<p>Chúng không có sai, nếu bạn chỉ quan tâm đến dự đoán.<sup><a name="r123" href="#123">123</a></sup> Vấn đề như sự kiên định luôn luôn được đánh giá theo <i>tiệm cận (asymptotically)</i>. Có nghĩa là chúng ta tưởng tượng cỡ mẫu đang lớn dần đến vô cực. Sau đó chúng ta đặt câu hỏi quy trình hoạt động như thế nào ở data vô hạn này. Với data hữu hạn thực tế, AIC và WAIC và cross-validation thường sẽ chọn mô hình phức tạp hơn, cho nên chúng đôi được cho là nguyên nhân của "overfitting". Nhưng ở data vô hạn, mô hình phức tạp nhất sẽ cho dự đoán y như mô hình thực (giả định nó tồn tại trong tập mô hình). Lý do ở đây là với rất nhiều data, mọi parameter được ước lượng cực kỳ chính xác. Và do đó khi sử dụng mô hình quá phức tạp sẽ không ảnh hưởng đến dự đoán. Ví dụ, khi cỡ mẫu $N \to \infty$, mô hình với 5 parameter trong hình 7.8 sẽ cho bạn hệ số của các biến dự đoán sau biến thứ hai là gần bằng zero. Cho nên việc thất bại xác định mô hình "đúng" sẽ không gây hại, ít nhất không phải nếu theo suy nghĩ này. Hơn nữa, trong khoa học tự nhiên và xã hội, mô hình đang tìm hiểu sẽ không bao giờ là mô hình tạo data. Cho nên không có lý gì để tìm ra mô hình "thực".</p>
</div>

<div class="alert alert-info">
    <p><strong>Còn BIC và Bayes factor thì sao?</strong> <strong>BAYESIAN INFORMATION CRITERION</strong>, gọi tắt là BIC và cũng được gọi là tiêu chuẩn Schwarz,<sup><a name="r124" href="#124">124</a></sup> thường được đặt kế bên AIC. Sự lựa chọn giữa AIC và BIC (hoặc cái khác!) không liên quan với có phải Bayes hay không. Chúng ta có cả phương pháp Bayes và non-Bayes để diễn giải chúng, và nếu cứng nhắc hơn, không cái nào là Bayes cả. BIC liên quan đến logarith của <i>likelihood trung bình</i> của mô hình tuyến tính. Likelihood trung bình là mẫu số trong Bayes' theorem, likelihood trung bình của prior. Một truyền thống đáng tôn trọng trong suy luận Bayes là so sánh likelihood trung bình như là một cách để so sánh mô hình. Tỉ số likelihood trung bình gọi là <strong>YẾU TỐ BAYES (BAYES FACTOR)</strong>. Ở thang đo logarith, những tỉ số này chính là hiệu số, và so sánh sự khác nhau giữa các likelihood trung bình có thể xem như so sánh sự khác nhau trong tiêu chuẩn thông tin. Vì likelihood trung bình được tính trên prior, nhiều parameter hơn tạo ra số hạng phạt đền tự nhiên dựa trên mức độ phức tạp. Nó giúp chúng ta tránh khỏi overfitting, mặc dù sự phạt đền chính xác không giống như tiêu chuẩn thông tin.</p>
    <p>Rất nhiều nhà thống kê Bayes không thích cách tiếp cận Bayes factor,<sup><a name="r125" href="#125">125</a></sup> họ thừa nhận rằng có vài cản trở về mặt kỹ thuật để sử dụng nó. Một lý do là việc tính likelihood trung bình rất khó. Ngay cả khi bạn có thể tính posterior, bạn không thể ước lượng được likelihood trung bình. Một vấn đề nữa là, ngay cả với prior yếu và không ảnh hưởng nhiều đến phân phối posterior giữa các mô hình, prior vẫn có thể có ảnh hưởng nhiều đến sự so sánh giữa các mô hình.</p>
    <p>Quan trọng hơn là nhận ra rằng, việc lựa chọn Bayes hay không, không liên quan đến lựa chọn tiêu chuẩn thông tin hay Bayes factor. Hơn nữa, ta thực sự không cần phải lựa chọn. Chúng ta có thể dùng cả hai và học từ kết quả đồng thuận hay không đồng thuận của chúng. Cả tiêu chuẩn thông tin và bayes factor đều là công cụ dự đoán đơn thuần mà có thể chọn những mô hình bị sai lệch (confounded). Chúng không biết gì về quan hệ nhân quả.</p>
</div>

<div class="alert alert-dark">
    <p><strong>Tính WAIC.</strong> Để biết được WAIC được tính như thế nào, ta xem ví dụ hồi quy đơn giản này:
    <pre><code>cars = pd.read_csv("https://github.com/fehiepsi/rethinking-numpyro/blob/master/data/cars.csv?raw=true", sep=",")

def model(speed, cars_dist):
    a = numpyro.sample("a", dist.Normal(0, 100))
    b = numpyro.sample("b", dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b * speed
    numpyro.sample("dist", dist.Normal(mu, sigma), obs=cars_dist)

m = AutoLaplaceApproximation(model)
svi = SVI(
    model, m, optim.Adam(1), Trace_ELBO(), speed=cars.speed.values, cars_dist=cars.dist.values
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
params = svi.get_params(state)
post = m.sample_posterior(random.PRNGKey(94), params, (1000,))</code></pre>
    Chúng ta phải cần thêm log-likelihood của mỗi quan sát $i$ của mỗi mẫu $s$ từ posterior:
    <pre><code>n_samples = 1000

def logprob_fn(s):
    mu = post["a"][s] + post["b"][s] * cars.speed.values
    return dist.Normal(mu, post["sigma"][s]).log_prob(cars.dist.values)

logprob = vmap(logprob_fn, out_axes=1)(jnp.arange(n_samples))</code></pre>
    Bạn có được một ma trận 50x1000 của log-likelihood, với quan sát ở các dòng và mẫu ở các cột. Để tính lppd, deviance của Bayes, chúng ta lấy trung bình các mẫu ở một dòng, lấy logarith, và cộng chúng lại.
    <pre><code>n_cases = cars.shape[0]
lppd = logsumexp(logprob, 1) - jnp.log(n_samples)</code></pre>
    Sau đó tính tổng của <code>lppd</code> sẽ cho kết quả lppd, như được nói ở bài chính. Đến phần số hạng phạt đền, $p_{\Tiny WAIC}$. Nó thì rõ ràng hơn, chúng ta chỉ cần tính phương sai giữa các mẫu cho mỗi quan sát, và cộng chúng lại:
    <pre><code>pWAIC = jnp.var(logprob, 1)</code></pre>
    Và tổng của <code>pWAIC</code> trả về $p_{\Tiny WAIC}$. Để tính WAIC:
    <pre><code>-2 * (jnp.sum(lppd) - jnp.sum(pWAIC))</code></pre></p>

    <p><samp>423.3154</samp></p>

    <p>So sánh với kết quả của hàm WAIC. Nó sẽ có độ biến thiên hay phương sai do mô phỏng, bởi vì mẫu được rút ra từ mô hình được fit. Nhưng phương sai sẽ rất nhỏ hơn so với sai số chuẩn của WAIC. Bạn có thể tính sai số chuẩn bằng cách tính căn bậc hai của số lượng các trường hợp nhân với phương sai của từng quan sát trong WAIC:
    <pre><code>waic_vec = -2 * (lppd - pWAIC)
jnp.sqrt(n_cases * jnp.var(waic_vec))</code></pre></p>

    <p><samp>17.81628</samp></p>

    <p>Khi mô hình càng phức tạp, tất cả những gì thay đổi thường là cách log-prob được tính.</p>
    <p>Cần ghi chú rằng mỗi quan sát cụ thể đêu có số hạng trừng phạt riêng trong vector <code>pWAIC</code> mà chúng ta tính ở trên. Nó cho phép chúng ta tìm hiểu những quan sát khác nhau đã đóng góp như thế nào cho overfitting.</p>
</div>

### 7.4.3 So sánh CV, PSIS, và WAIC

Với những định nghĩa về cross-validation, PSIS, và WAIC, ta hãy làm một thí nghiệm. Nó sẽ giúp ta vẽ biểu đồ của những ước lượng deviance ngoài mẫu theo những phương pháp trên, với bối cảnh tương tự như phần trước. Mục đích của chúng ta là xem những phương pháp trên sẽ ước lượng độ chính xác ngoài mẫu tốt như thế nào. Chúng có ước lượng đúng overfitting?

![](/assets/images/fig 7-9.svg)
<details class="fig"><summary>Hình 7.9: WAIC và cross-validation đều là ước lượng của deviance ngoài mẫu. Hàng trên cho 100 mô phỏng huấn luyện-kiểm tra với $N=20$. Hàng dưới cho 100 mô phỏng với $N=100$. Ở mỗi biểu đồ có hai xu thế. Những điểm màu trắng là prior không được regularize. Những điểm màu xanh là prior được regularize $\sigma=0.5$. Bên trái: Trục tung là deviance tuyệt đối. Những điểm tròn là trung bình của deviance kiểm tra. Đường đen là ước lượng WAIC trung bình. Màu xanh là điểm leave-one-out cross-validation, và đường xanh nét đứt là ước lượng PSIS của điểm cross-validation. Bên phải: cùng một data, nhưng được tính theo sai số trung bình của ước lượng của deviance kiểm tra. </summary>
    <pre><code></code>def compare_CV_PSIS_WAIC(N, k, b_sigma, i):
    # create sample
    x_train, y_train = sim_train_test(N, k, i, rng_key=0)
    x_test, y_test = sim_train_test(N, k, i, rng_key=1)
    post = fit(model, x_train, y_train, b_sigma, i, rng_key=2, scalar=False)
    if k > 1:
        coefs = jnp.concatenate([
            post["a"][0],
            jnp.median(post['b'], axis=0)
        ])
        logprob = log_likelihood(model, post, x_train, y_train, b_sigma)
        az_data = az.from_dict(
            posterior={'b': post['b'].reshape(1, 1000, k-1)},
            log_likelihood={'y':logprob['y'].reshape(1, 1000, N)}
        )
    else:
        coefs = post["a"][0]
        logprob = dist.Normal(post["a"]).log_prob(y_train)
        az_data = az.from_dict(
            posterior={'a': post['a'].reshape(1, 1000, 1)},
            log_likelihood={'y':logprob.reshape(1, 1000, N)})
    
    mu = jnp.matmul(x_test, coefs)
    logprob = dist.Normal(mu).log_prob(y_test)
    true_dev = (-2) * jnp.sum(logprob)
    # CV
    picked_rows_list = [np.delete(np.arange(N),idx) for idx in range(N)]
    x_train_list = vmap(lambda rows: x_train[rows,:k])(jnp.array(picked_rows_list))
    y_train_list = vmap(lambda rows: y_train.reshape(-1,1)[rows,:])(jnp.array(picked_rows_list))
    xy_train_list = jnp.concatenate([x_train_list, y_train_list], axis=2)
    coefs_list = vmap(
        lambda xy: fit(model, xy[:,:-1], xy[:,-1], b_sigma, i, rng_key=3))(xy_train_list)
    excluded_x_train_list = vmap(lambda row: x_train[row,:k])(jnp.arange(N))
    excluded_y_train_list = vmap(lambda row: y_train[row])(jnp.arange(N))
    mus = jnp.sum(coefs_list* excluded_x_train_list, axis=1)
    logprob = dist.Normal(mus).log_prob(excluded_y_train_list)
    CV = jnp.sum(((-2) * logprob))
    # LOO
    loo_score = az.loo(az_data, scale='deviance').loo
    # WAIC
    waic_score = az.waic(az_data, scale='deviance').waic
    return jnp.stack([true_dev, CV, loo_score, waic_score])
    
def compare_fn(N, k, b_sigma, num=1e2):
    '''
    return 8 values:
    mean: true_dev, CV, loo, waic
    std : true_dev, CV, loo, waic
    '''
    r = jnp.array([])
    for i in jnp.arange(int(num)):
        print(i, end='--')
        xx = compare_CV_PSIS_WAIC(N, k, b_sigma, i)
        r = jnp.append(r, xx)
    print(f"end iterating {k} parameter")
    r = r.reshape(int(num), 4)
    return jnp.concatenate([jnp.mean(r, 0), jnp.std(r, 0)])
fig,ax = plt.subplots(2, 2,figsize=(12,10))
kseq = range(1, 6)

for idx, N in  in enumerate([20,100]): 
    comp1 = jnp.stack([compare_fn(N, k, b_sigma=100) for k in kseq], axis=1)
    comp2 = jnp.stack([compare_fn(N, k, b_sigma=0.5) for k in kseq], axis=1)
    ax[idx, 0].scatter(kseq, comp1[0], s=200, facecolor="w", edgecolor="b")
    ax[idx, 0].plot(kseq, comp1[1], c='b', label="CV")
    ax[idx, 0].plot(kseq, comp1[2], c='b', ls="--", label="PSIS")
    ax[idx, 0].plot(kseq, comp1[3], c='k', label="WAIC")
    ax[idx, 0].set_title(f"N={N}")
    ax[idx, 0].legend()
    
    ax[idx, 1].plot(kseq, comp1[5], c='b')
    ax[idx, 1].plot(kseq, comp1[6], c='b', ls="--")
    ax[idx, 1].plot(kseq, comp1[7], c='k')
    ax[idx, 1].set_title(f"N={N}")
    ax[idx, 1].annotate("Flat prior", (5, comp1[7,4]))
    
    ax[idx, 0].scatter(kseq, comp2[0], s=200)
    ax[idx, 0].plot(kseq, comp2[1], c='b')
    ax[idx, 0].plot(kseq, comp2[2], c='b', ls="--")
    ax[idx, 0].plot(kseq, comp2[3], c='k')
    ax[idx, 0].set_title(f"N={N}")
    
    ax[idx, 1].plot(kseq, comp2[5], c='b')
    ax[idx, 1].plot(kseq, comp2[6], c='b', ls="--")
    ax[idx, 1].plot(kseq, comp2[7], c='k')
    ax[idx, 1].set_title(f"N={N}")
    ax[idx, 1].annotate("sigma=0.5", (5, comp2[7,4]))
plt.tight_layout()</pre></details>

Hình 7.9 cho kết quả của 100 mô phỏng cho mỗi mô hình trong năm mô hình quen thuộc giữa 1 và 5 parameter, được mô phỏng dưới hai tập prior khác nhau và hai cỡ mẫu khác nhau. Hình này phức tạp, những hãy xem từng mảnh một, chúng ta đều đã quen thuộc với chúng. Hãy tập trung vào chỉ hình trên bên trái, với $N=20$. Trục tung là deviance ngoài mẫu ($-2 \times \text{lppd}$). Những điểm trắng là deviance trung bình ngoài mẫu với mô hình có prior phẳng. Những điểm xanh là deviance trung bình ngoài mẫu có prior được regularize với độ lệch chuẩn 0.5. Chú ý rằng prior được regularize thì bị ít overfit hơn, cũng giống như những gì bạn thấy ở phần trước. Không có gì mới.

Chúng ta bây giờ sẽ quan tâm đến CV, PSIS, WAIC ước lượng những điểm này tốt như thế nào. Vẫn nhìn vào hình trên bên trái, ta có những đường thẳng cho mỗi phương pháp ước lượng. Đường đen nét liền là WAIC. Đường xanh nét liền là cross-validation được thực hiện trực tiếp, bằng cách fit mô hình $N$ lần. Đường xanh nét đứt cho PSIS. Chú ý rằng tất cả phương pháp đều rất tốt trong việc ước lượng điểm trung bình ngoài mẫu, do dù là prior phẳng (trên) hay prior hẹp (dưới). Cho rằng mô hình xử lý tạo data là không đổi, ta thực sự có thể dùng một mẫu đơn độc để ước lượng độ chính xác của dự đoán.

Trong khi cả 3 phương pháp đều cho ước lượng deviance ngoài mẫu khá chính xác, nó cũng đúng nếu có một mẫu nào khác mà chúng bị sai. Cho nên chúng ta phải nhìn vào sai số trung bình. Biểu đồ trên bên phải cho thấy sai số trung bình cho mỗi phương pháp ước lượng. Trục tung là sai số tuyệt đối trung bình giữa deviance ngoài mẫu và mỗi phương pháp. WAIC (màu đen) cho kết quả tốt hơn. Biểu đồ ở hàng dưới là biểu đồ cho cỡ mẫu lớn hơn, $N=100$. Với cỡ mẫu lớn hơn, trong mô hình đơn giản này, cả ba phương pháp đều như nhau.

PSIS và WAIC hoạt động rất giống nhau trong trường hợp dùng hồi quy tuyến tính thông thường.<sup><a name="r126" href="#126">126</a></sup> Nếu có sự khác biệt nào quan trọng, nó nằm ở loại mô hình khác nhau, khi mà phân phối posterior không phải Gaussian và có sự hiện diện có quan sát ảnh hưởng mạnh đến posterior. CV và PSIS có phương sai lớn hơn khi ước lượng độ phân kỳ KL, trong khi WAIC bị sai số lớn hơn. Cho nên chúng ta phải lường trước được phương pháp nào tốt hơn trong bối cảnh nào.<sup><a name="r127" href="#127">127</a></sup> Tuy nhiên, trong thực hành sự ưu thế hơn có thể nhỏ hơn nhiều với sai số mong đợi. Watanabe khuyến nghị tính cả WAIC và PSIS và so sánh chúng. Nếu như có khác biệt quá lớn, điều đó có nghĩa một trong hai phương là không đáng tin cậy.

Ngoài việc ước lượng, PSIS có lợi thế đặc biệt hơn ở việc cảnh báo người dùng khi mà có gì đó bất ổn. Giá trị $k$ trong PSIS cho mỗi quan sát chỉ điểm cho việc điểm PSIS không đáng tin cậy, cũng như xác định quan sát nào đó là nguyên nhân. Chúng ta sẽ xem lợi thế này rõ hơn ở phần sau.

<div class="alert alert-info">
    <p><strong>Nhiều khung quy trình thực hiện dự đoán.</strong> Quy trình huấn luyện kiểm tra ở trên giả định mẫu kiểm tra có cùng cỡ mẫu và đặc tính như mẫu huấn luyện. Và điều đó không có nghĩa là tiêu chuẩn thông tin chỉ được dùng khi chúng ta dự đoán mẫu cùng cỡ mẫu với mẫu huấn luyện. Việc cùng cỡ mẫu là để cho deviance ngoài mẫu gần như nhau. Quan trọng hơn là khoảng cách giữa các mô hình, không phải giá trị thực sự của deviance. Cả cross-validation và tiêu chuẩn thông tin đều không cần mô hình tạo data thực sự. Nó đúng trong ví dụ của chúng ta. Nhưng nó không phải điều kiện cần để chúng giúp ta phát hiện mô hình tốt cho dự đoán.</p>
    <p>Nhưng công việc dự đoán huấn luyện-kiểm tra không đại diện cho tất cả chúng ta muốn làm với mô hình. Ví dụ, một vài nhà thống kê thích đánh giá dự đoán dựa vào khung quy trình <strong>PREQUENTIAL</strong>, trong đó mô hình được đánh giá dựa vào sai số học tích luỹ trên mẫu huấn luyện.<sup><a name="r128" href="#128">128</a></sup> Và khi bạn học đến mô hình đa tầng, "dự đoán" không còn được định nghĩa độc nhất nữa, bởi vì mẫu kiểm tra có thể khác với mẫu huấn luyện mà không cho dùng ước lượng vài parameter. Ta sẽ gặp lại vấn đề này ở Chương 13.</p>
    <p>Có lẽ lo lắng lớn nhất là thực nghiệm huấn luyện kiểm tra của chúng ta lấy mẫu kiểm tra đúng như mô hình xử lý của mẫu huấn luyên. Đây là một loại giả định <i>đồng nhất</i>, tức là data tương lai được dự kiến đến từ chung một mô hình xử lý như data cũ và có một khoảng giá trị nhất định. Điều này có thể gây trục trặc. Ví dụ, giả sử chúng ta fit mô hình tuyến tính dự đoán chiều cao bằng cân nặng. Mẫu huấn luyện đến từ một thành phố nghèo, đa số đều rất ốm. Quan hệ giữa chiều cao và cân nặng sẽ rất lớn. Bây giờ giả sử nhiệm vụ của bạn là dự đoán chiều cao ở một nơi khác giàu hơn. Việc đưa cân nặng ở những người giàu hơn vào mô hình dành cho người nghèo, sẽ dự đoán ra những người cực kỳ cao. Lý do là, một khi cân nặng đủ lớn, nó không có quan hệ gì với chiều cao. WAIC sẽ không tự động phát hiện hay giải quyết vấn đề này. Cũng như những quy trình khác. Nhưng việc lặp lại fit mô hình, mục tiêu để dự đoán, hay phê phán mô hình, ta có thể vượt qua được giới hạn loại này. Thống kê mãi mãi không thay thế được khoa học.</p>
</div>

## <center>7.5 So sánh mô hình</center><a name="a5"></a>

Hãy xem lại câu hỏi ban đầu và con đường đã đi đến đây. Khi có nhiều mô hình được ứng cử (hi vọng không bị sai lệch - confound) cho cùng một dữ liệu, thì chúng ta làm thế nào để so sánh độ chính xác của những mô hình đó? Việc chạy theo mức độ fit mô hình là sai, bởi vì mức độ fit luôn tăng theo sự phức tạp của mô hình. Độ phân kỳ thông tin là lựa chọn đúng để đánh giá độ chính xác mô hình, nhưng ngay cả nó cũng có thể dẫn chúng ta đến việc chọn mô hình ngày càng phức tạp hơn và có thể là mô hình sai. Chúng ta cần phải biết đánh giá mô hình dựa trên dữ liệu ngoài mẫu. Làm sao làm được chuyện đó? Một mô hình gộp cho ta biết hai thứ quan trọng. Đầu tiên, prior phẳng tạo ra dự đoán kém. Regularizing prior - những prior đa nghi với giá trị parameter cực lớn hoặc cực nhỏ - giúp giảm thiểu mức độ fit mô hình nhưng cải thiện độ chính xác dự đoán. Thứ hai, chúng ta có thể đoán được độ chính xác dự đoán thông qua CV, PSIS, WAIC. Regularizing prior và CV/PSIS/WAIC là hỗ trợ lẫn nhau. Regularizing sẽ giúp giảm overfitting, và CV/PSIS/WAIC đo đạc nó.

Đây là con đường đã đi, con đường lý thuyết. Và nó là phần khó xơi nhất. Sử dụng công cụ PSIS và WAIC thì dễ rất nhiều so với việc hiểu chúng. Do đó chúng khá nguy hiểm. Đó là lý do tại sao chương này dành nhiều thời gian cho kiến thức cơ sở, mà không thực hiện phân tích data thực thụ.

Bây giờ ta sẽ phân tích dự liệu. Làm sao sử dụng regularizing prior và CV/PSIS/WAIC? Một ứng dụng thường gặp của cross-validation và tiêu chuẩn thông tin là thực hiện **CHỌN LỌC MÔ HÌNH (MODEL SELECTION)**, tức là chọn mô hình nào có giá trị ước lượng thấp nhất và từ bỏ những mô hình còn lại. Nhưng bạn không nên làm vậy. Quy trình chọn lọc này từ bỏ những thông tin về độ chính xác tương đối của mô hình nằm ở sự khác nhau giữa các giá trị CV/PSIS/WAIC. Tại sao sự khác nhau đó có ích? Bởi vì đôi khi sự khác nhau này rất lớn hoặc rất nhỏ. Cũng giống như phân phối posterior nói về mức độ tin cậy của parameter (đặt điều kiện trên mô hình), độ chính xác tương đối của mô hình cho ta biết mức độ tin cậy mô hình của chúng ta (đặt điều kiện trên tập mô hình dùng để so sánh).

Một lý do khác không bao giờ chọn mô hình chỉ dựa vào WAIC/CV/PSIS là chúng ta đôi khi cần quan tâm đến quan hệ nhân quả. Tối đa hoá độ chính xác dự đoán là không giống như suy luận nhân quả. Mô hình bị confound vẫn có thể cho dự đoán tốt, nhất là trong tương lai gần. Chúng không nói cho ta biết hệ quả của một can thiệp, nhưng nó giúp ta dự đoán. Cho nên chúng ta cần phải rõ ràng về mục tiêu và không nên thêm biến số bừa bãi vào salad nhân quả và để WAIC lựa chọn bữa ăn của chúng ta.

Vậy những tiêu chuẩn đó có gì hay? Chúng đo đạc giá trị dự đoán mong đợi của biến số ở trên cùng một cân đo, giải thích cho overfitting. Nó giúp chúng ta kiểm tra những kết quả từ mô hình, dưới sự hướng dẫn của mô hình nhân quả. Chúng cũng cung cấp một cách đo đạc khuynh hướng overfitting của mô hình, và nó giúp chúng ta thiết kế mô hình cũng như hoạt động của suy luận thống kê. Cuối cùng, tối thiểu hoá tiêu chuẩn như WAIC giúp chúng ta thiết kê mô hình, đặc biệt là tinh chỉnh parameter trong mô hình đa tầng. 

Cho nên thây vì *chọn lựa* mô hình, chúng ta tập trung vào **SO SÁNH MÔ HÌNH (MODEL COMPARISON)**. Nó là một cách tiếp cận tổng quát mà sử dụng nhiều mô hình để hiểu hai vấn đề, biến khác nhau ảnh hưởng thế nào đến dự đoán, và khi kết hợp với mô hình nhân quả, suy ra các mối quan hệ độc lập có điều kiện giữa các biến, giúp ta suy luận nhân quả.

Chúng ta sẽ học qua hai ví dụ. Ví dụ thứ nhất nhấn mạnh sự khác biệt giữa các mô hình đang so sánh, so sánh hiệu năng của chúng trong tình huống dự đoán và suy luận nhân quả. Ví dụ thứ hai nhấn mạnh tính chất từng điểm (pointwise) của việc so sánh mô hình và việc kiểm tra giá trị quan sát cụ thế cho thấy hiệu năng mô hình và sai lệch chuyên biệt. Ví dụ thứ hai cũng giới thiệu một phương pháp thay thế mạnh hơn (robust) cho hồi quy Gaussian.

### 7.5.1 Chọn sai mô hình

Chúng ta nhớ lại những vài học ở chương trước: Suy luận nhân quả và dự đoán là hai tác vụ khác nhau. Cross-validation và WAIC nhắm tới tìm mô hình cho dự đoán tốt. Chúng không giải quyết vấn đề nhân quả. Nếu bạn chọn mô hình chỉ dựa trên độ chính xác dự đoán mong đợi, bạn sẽ dễ bị confound. Lý do là những backdoor vẫn cho chúng ta thông tin về quan hệ thống kê trong data. Cho nên chúng cải thiện dự đoán, khi và chỉ khi chúng ta không can thiệp vào hệ thống và tương lai giống như quá khứ. Nhưng nhớ lại rằng định nghĩa nhân quả là chúng ta có thể dự đoán hậu quả của một can thiệp. Cho nên một con số PSIS hoặc WAIC đẹp không đồng nghĩa với mô hình nhân quả.

Ví dụ, nhớ lại ví dụ trong cây ở chương trước. Mô hình có đặt điều kiện lên biến nấm sẽ cho kết qur dự đoán tốt hơn mô hình không có nó. Nếu bạn trở lại phần trước và chạy mô hình `m6.6`, `m6.7`, `m6.8`, chúng ta có thể so sánh giá trị WAIC của chúng. Nhắc lại, `m6.6` là mô hình với chỉ intercept, `m6.7` là mô hình có biến điều trị và nấm (biến hậu điều trị), và `m6.8` là mô hình chứa cả điều trị và không có biến nấm. Mô hình `m6.8` là mô hình suy luận nhân quả đúng về hiệu ứng của điều trị.

Để bắt đầu, ta sẽ tính WAIC của `m6.7`:

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


def model(h0, h1):
    p = numpyro.sample("p", dist.LogNormal(0, 0.25))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = h0 * p
    numpyro.sample("h1", dist.Normal(mu, sigma), obs=h1)


m6_6 = AutoLaplaceApproximation(model)
svi = SVI(model, m6_6, optim.Adam(0.1), Trace_ELBO(), h0=d.h0.values, h1=d.h1.values)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p6_6 = svi.get_params(state)


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

post = m6_7.sample_posterior(random.PRNGKey(11), p6_7, (1000,))
logprob = log_likelihood(
    m6_7.model,
    post,
    treatment=d.treatment.values,
    fungus=d.fungus.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
az6_7 = az.from_dict(sample_stats={"log_likelihood": logprob["h1"][None, ...]})
az.waic(az6_7, scale="deviance")
```

<p><samp><table border="1">
<thead>
<tr style="text-align: right;">
    <th></th>
    <th>Estimate</th>
    <th>SE</th>
</tr>
</thead>
<tbody>
<tr>
  <th>deviance_waic</th>
  <td>336.02</td>
  <td>14.65</td>
</tr>
<tr>
  <th>p_waic</th>
  <td>4.08</td>
</tr>
</tbody>
</table></samp></p>

Giá trị đầu tiên là giá trị ước lượng deviance ngoài mẫu. Những giá trị khác là sai số chuẩn của WAIC, số lượng parameter hiệu quả. Nội dung ở phần trước đã nói cách tính những con số đó. Để so sánh nhiều mô hình đơn giản hơn, ta có thể dùng hàm `compare` trong `arviz`:

```python
post = m6_6.sample_posterior(random.PRNGKey(77), p6_6, (1000,))
logprob = log_likelihood(m6_6.model, post, h0=d.h0.values, h1=d.h1.values)
az6_6 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
post = m6_7.sample_posterior(random.PRNGKey(77), p6_7, (1000,))
logprob = log_likelihood(
    m6_7.model,
    post,
    treatment=d.treatment.values,
    fungus=d.fungus.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
az6_7 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
post = m6_8.sample_posterior(random.PRNGKey(77), p6_8, (1000,))
logprob = log_likelihood(
    m6_8.model, post, treatment=d.treatment.values, h0=d.h0.values, h1=d.h1.values
)
az6_8 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
az.compare({"m6.6": az6_6, "m6.7": az6_7, "m6.8": az6_8}, ic="waic", scale="deviance")
```

<p><samp><table border="1">
<thead>
<tr style="text-align: right;">
    <th></th>
    <th>rank</th>
    <th>waic</th>   
    <th>p_waic </th>
    <th>d_waic </th>
    <th>weight </th>
    <th>se </th>
    <th>dse</th>
    <th>warning</th>
    <th>waic_scale</th>
</tr>
</thead>
<tbody>
<tr>
    <th>m6.7</th>
    <td>0  </td>
    <td>335.903</td>
    <td>4.01345</td>
    <td>0  </td>
    <td>0.99977</td>
    <td>16.0559</td>
    <td>0  </td>
    <td>True   </td>
    <td>deviance</td>
</tr>
<tr>
    <th>m6.8</th>
    <td>1</td>
    <td>399.758</td>
    <td>3.08942</td>
    <td>63.8551</td>
    <td>0.000229738</td>
    <td>12.9573</td>
    <td>15.1241</td>
    <td>True</td>
    <td>deviance</td>
</tr>
<tr>
    <th>m6.6</th>
    <td>2</td>  
    <td>409.201</td>
    <td>1.71209</td>
    <td>73.2974</td>
    <td>4.42385e-08</td>
    <td>14.2718</td>
    <td>14.3947</td>
    <td>False</td>
    <td>deviance</td>
</tr></tbody>
</table></samp></p>

Kết quả PSIS cũng sẽ như nhau. Bạn có thể dùng `ic="psis"` trong hàm `compare` để kiểm tra. Mỗi dòng là một mô hình. Cột từ trái sang phải là: xếp hạng, WAIC, số lượng parameter hiệu quả, khoảng cách đến WAIC tốt nhất, trọng số, sai số chuẩn của WAIC, sai số chuẩn của khoảng cách, cảnh báo, thang đo. Mỗi giá trị này đều cần nhiều giải thích.

Giá trị WAIC ở cột thứ hai. Giá trị càng nhỏ càng tốt, và mô hình được xếp hạng từ mô hình tốt nhất đến tệ nhất. Mô hình có biến nấm thì có WAIC nhỏ nhất, như đã nói. `pWAIC` là số hạng phạt đền của WAIC. Những giá trị này thì gần bằng, nhưng hơi nhỏ hơn, số chiều của posterior ở mỗi mô hình, điều đã được mong đợi ở mô hình hồi quy tuyến tính với regularizing prior. Số hạng phạt đền sẽ hấp dẫn hơn ở phần sau của sách.

Cột `dWAIC` là sự khác nhau giữa mỗi mô hình và mô hình tốt nhất. Nó bằng zero ở mô hình tốt nhất và hiệu của những mô hình khác nói bạn biết khoảng cách đến mô hình ở đầu. Vậy `m6.7` là nhỏ hơn khoảng 60 đơn vị deviance so với mô hình khác. Mô hình intercept, `m6.6` tệ hơn `m6.8` 10 đơn vị. Những hiệu số này khác nhau nhiều hay ít? Một cách để trả lời câu hỏi này là hỏi rõ ràng hơn: Những mô hình có khác nhau bởi độ chính xác dự đoán ngoài mẫu của chúng? Để trả lời câu hỏi, bạn phải xem xét sai số của ước lượng WAIC. Bởi vì chúng ta không biết mẫu mục tiêu, đây chỉ là những ước đoán, và chúng ta biết rằng trong mô phỏng có nhiều biến thiên ở sai số của WAIC.

Và đó là hai cột `se` và `dse`, sẽ giúp chúng ta. `se` là sai số chuẩn của mỗi WAIC. Có thể hiểu rằng, tính bất định của độ chính xác ngoài mẫu được phân phối theo phân phối chuẩn với trung bình là số WAIC và độ lệch chuẩn bằng sai số chuẩn. Khi cỡ mẫu nhỏ, ước lượng tính bất định này có khuynh hướng biến thiên hơn. Nhưng nó vẫn tốt hơn tiêu chuẩn cũ như AIC, vì nó không cho ta cách để đánh giá tính bất định.

Bây giờ ta đánh giá mức độ tin cậy sự khác nhau của mô hình, chúng ta không dùng sai số chuẩn WAIC mà dùng sai số chuẩn của khoảng cách. Nghĩa là sao? Giống như mỗi WAIC, mỗi hiệu số của WAIC cũng có sai số chuẩn. Để tính toán sai số chuẩn của khoảng cách giữa mô hình `m6.7` và `m6.8`, ta chỉ cần thêm tính từng điểm của các giá trị WAIC:

```python
post = m6_7.sample_posterior(random.PRNGKey(91), p6_7, (1000,))
logprob = log_likelihood(
    m6_7.model,
    post,
    treatment=d.treatment.values,
    fungus=d.fungus.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
az6_7 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_7 = az.waic(az6_7, pointwise=True, scale="deviance")
post = m6_8.sample_posterior(random.PRNGKey(91), p6_8, (1000,))
logprob = log_likelihood(
    m6_8.model, post, treatment=d.treatment.values, h0=d.h0.values, h1=d.h1.values
)
az6_8 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_8 = az.waic(az6_8, pointwise=True, scale="deviance")
n = waic_m6_7.n_data_points
diff_m6_7_m6_8 = waic_m6_7.waic_i.values - waic_m6_8.waic_i.values
jnp.sqrt(n * jnp.var(diff_m6_7_m6_8))
```

<samp>15.060513</samp>

Đây là giá trị của hàng hai ở cột `dse`. Nó có thể hơi khác do biến thiên của mô phỏng. Sự khác nhau giữa mô hình là 63.8, và độ lệch chuẩn là 15. Nếu chúng ta tưởng tượng khoảng 99% của sự khác nhau (z-score khoảng 2.6), thì nó sẽ là:

```python
40.0 + jnp.array([-1, 1]) * 10.4 * 2.6
```

<samp>[12.960003, 67.03999]</samp>

Đúng vậy, những mô hình này có độ chính xác ngoài mẫu khác nhau đáng tin cậy. Mô hình `m6.7` thì tốt hơn nhiều. Bạn có thể nhìn rõ hơn trên biểu đồ:

```python
compare = az.compare(
    {"m6.6": az6_6, "m6.7": az6_7, "m6.8": az6_8}, ic="waic", scale="deviance"
)
az.plot_compare(compare)
```

![](/assets/images/plot_compare.svg)

Màu đen là deviance trong mẫu, còn mà xanh là WAIC. Để ý rằng nhìn chung mô hình sẽ làm tốt hơn với dữ liệu trong mẫu hơn là ngoài mẫu. Đường thẳng cho sai số chuẩn của mỗi WAIC. Đó cũng là giá trị ở cột `se`. Bạn có thể thấy `m6.7` tốt hơn như thế nào so với `m6.8`. Chúng ta cần thêm là sai số chuẩn của khoảng cách WAIC giữa hai mô hình. Đó là đường thẳng nằm trên với hình tam giác, nằm giữa `m6.7` và `m6.8`.

Điều đó nghĩa là gì? Nghĩa là WAIC không dùng để suy luận nhân quả. Chúng ta biết sự thật, bởi vì chúng ta là người mô phỏng ra data, là điều trị có ý nghĩa.  Những bởi vì nấm đã nằm trung gian trên đường đi của điều trị (mediation), khi chúng ta đặt điều kiện lên nấm, điều trị sẽ không cho thông tin nào thêm. Và bởi vì nấm tương quan nhiều hơn với kết cục, mô hình sử dụng nó sẽ cho dự đoán tốt hơn. WAIC đã hoàn thành công việc của nó. Công việc của nó không phải suy luận thống kê. Công việc của nó là ước lượng độ chính xác dự đoán. 

Không có nghĩa là WAIC (hay CV hay PSIS) là vô dụng. Nó cho ta một đại lượng có ích để cải thiện dự đoán từ việc đặt điều kiện lên nấm. Mặc dù điều trị có hiệu quả, nó không phải luôn luôn là 100%, cho nên việc biết điều trị sẽ không thay thế được việc biết có hiện diện nấm.

Tương tự, chúng ta có thể hỏi thêm rằng sự khác nhau giữa mô hình `m6.8`, mô hình với chỉ điều trị, và mô hình `m6.6`, mô hình chỉ intercept. Mô hình `m6.8` cho ta bằng chứng tốt là điều trị có hiệu quả. Bạn có thể kiểm tra lại posterior nếu bạn quên. Nhưng WAIC nghĩ hai mô hình này khác giống nhau. Sự khác nhau của chúng chỉ có 10 đơn vị deviance. Hãy tính sai số chuẩn khoảng cách để làm rõ hơn:

```python
post = m6_6.sample_posterior(random.PRNGKey(92), p6_6, (1000,))
logprob = log_likelihood(m6_6.model, post, h0=d.h0.values, h1=d.h1.values)
az6_6 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_6 = az.waic(az6_6, pointwise=True, scale="deviance")
diff_m6_6_m6_8 = waic_m6_6.waic_i.values - waic_m6_8.waic_i.values
jnp.sqrt(n * jnp.var(diff_m6_6_m6_8))
```

<samp>7.524173</samp>

Bảng so sánh không cho giá trị này, mặc dù có tính nó. Ta có thể làm bảng ma trận với toàn bộ mô hình và các giá trị độ lệch chuấn khoảng cách giữa chúng.

```python
post = m6_6.sample_posterior(random.PRNGKey(93), p6_6, (1000,))
logprob = log_likelihood(m6_6.model, post, h0=d.h0.values, h1=d.h1.values)
az6_6 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_6 = az.waic(az6_6, pointwise=True, scale="deviance")
post = m6_7.sample_posterior(random.PRNGKey(93), p6_7, (1000,))
logprob = log_likelihood(
    m6_7.model,
    post,
    treatment=d.treatment.values,
    fungus=d.fungus.values,
    h0=d.h0.values,
    h1=d.h1.values,
)
az6_7 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_7 = az.waic(az6_7, pointwise=True, scale="deviance")
post = m6_8.sample_posterior(random.PRNGKey(93), p6_8, (1000,))
logprob = log_likelihood(
    m6_8.model, post, treatment=d.treatment.values, h0=d.h0.values, h1=d.h1.values
)
az6_8 = az.from_dict({}, log_likelihood={"h1": logprob["h1"][None, ...]})
waic_m6_8 = az.waic(az6_8, pointwise=True, scale="deviance")
dSE = lambda waic1, waic2: jnp.sqrt(
    n * jnp.var(waic1.waic_i.values - waic2.waic_i.values)
)
data = {"m6.6": waic_m6_6, "m6.7": waic_m6_7, "m6.8": waic_m6_8}
pd.DataFrame(
    {
        row: {col: dSE(row_val, col_val) for col, col_val in data.items()}
        for row, row_val in data.items()
    }
)
```

<p><samp><table border="1">
<thead>
<tr style="text-align: right;">
    <th></th>
    <th>m6.6</th>
    <th>m6.7</th>
    <th>m6.8</th>
</tr>
</thead>
<tbody>
<tr>
    <th>m6.6</th>
    <td>0.0</td>
    <td>14.389981</td>
    <td>7.558166</td>
</tr>
<tr>
    <th>m6.7</th>
    <td>14.389981</td>
    <td>0.0</td>
    <td>15.01256</td>
</tr>
<tr>
    <th>m6.8</th>
    <td>7.558166</td>
    <td>15.01256</td>
    <td>0.0</td>
</tr></tbody>
</table></samp></p>

Ma trận này chứa toàn bộ sai số chuẩn khoảng cách từng cặp mô hình. Chú ý rằng sai số chuẩn khoảng cách của `m6.6` và `m6.8` lớn hơn khoảng cách của chúng. Chúng ta không thể nào phân bité hai mô hình này dựa trên WAIC. Những hiệu số này có thế ít đáng tin cậy hơn so với sai số chuẩn của mỗi mô hình. Hiện tại vẫn chưa có phân tích nào ở những con hiệu số này, nhưng nó sẽ sớm xuất hiện.<sup><a name="r129" href="#129">129</a></sup>

Vậy có nghĩa là điều trị không hiệu quả? Dĩ nhiên không. Chúng ta đã biết nó hiệu quả. Chúng ta đã mô phỏng data. Và phân phối posterior của hiệu ứng điều trị, `bt` trong `m6.8`, là con số dương đáng tin cậy. Những nó không lớn lắm. Cho nên nó không cải thiện nhiều dự đoán chiều cao cây. Còn có nhiều nguồn biến thiên khác. Kết quả này chỉ nhắc lại sự thật rằng WAIC (và CV và PSIS): chúng ước lượng độ chính xác dự đoán, không phải suy luận nhân quả. Một biến số có thêm quan hệ nhân quả với kết cục, nhưng có hiệu ứng rất nhỏ tác động lên nó, và WAIC sẽ thể hiện điều đó. Đó là những gì xảy ra trong trường hợp này. Chúng ta có thể sử dụng WAIC/CV/PSIS để đo lường sự khác nhau lớn như thế nào khi điều chỉnh các biến số. Nhưng chúng ta không nên dùng nó để quyết định sự tồn tại của một hay nhiều hiệu ứng. Chúng ta cần phân phối posterior của nhiều mô hình, có thể kiểm tra mỗi quan hệ độc lập có điều kiện của sơ đồ nhân quả, để thực hiện điều đó.

Nhân tố cuối cùng của bảng `compare` là cột trọng số `weight`. Những giá trị này là cách cổ điển để tổng kết điểm số tương đối của mỗi mô hình. Chúng luôn có tổng là 1, trong một tập các mô hình cần so sánh. Trọng số của mô hình được tính theo:

$$ w_i = \frac{ \exp(-0.5 \Delta_i)} {\sum_j \exp(-0.5\Delta_j)} $$

trong đó $\Delta_i$ là hiệu WAIC giữa mô hình $i$ và mô hình tốt nhất trong tập so sánh. Nó là `dWAIC` trong bảng. Những trọng số này giúp nhìn nhanh độ lớn của sự khác nhau giữa các mô hình. Nhưng bạn có thể phải kiểm tra lại sai số chuẩn. Bởi vì trọng số không phản ánh sai số chuẩn. chúng đơn thuần không đủ cho so sánh mô hình. Trọng số cũng được dùng cho **TRUNG BÌNH HOÁ MÔ HÌNH (MODEL AVERAGING)**. Trung bình hoá mô hình là một nhánh phương pháp để kết hợp dự đoán của nhiều mô hình. Bạn có thể đọc thêm phần endnote để hiểu biết thêm.<sup><a name="r130" href="#130">130</a></sup>

<div class="alert alert-info">
    <p><strong>Minh hoạ ẩn dụ cho WAIC.</strong> Đây là hai minh hoạ ẩn dụ giúp giải thích thêm nguyên tắc đằng sau WAIC (hoặc tiêu chuẩn thông tin khác) trong so sánh mô hình.</p>
    <p>Hãy nghĩ nó như đua ngựa. Trong một cuộc thi, con ngựa tốt nhất có thể không thẳng được. Nhưng nó có khả năng thắng cao hơn con ngựa kém nhất. Và khi con ngựa thắng cuộc kết thúc trong nửa thời gian của con ngựa thắng thứ hai, bạn chắc chắn rằng con ngựa thắng cuộc là con ngựa tốt nhất. Nhưng nếu đó là một kết thúc trong tích tắc, gần như là hoà cuộc giữa con ngựa về nhất và về nhì, thì độ tự tin của bạn cũng giảm đi về con ngựa tốt nhất. WAIC cũng tương tự như thời gian trong cuộc đua - giá trị nhỏ hơn thì tốt hơn, và khoảng cách giữa con ngựa/mô hình cũng tồn tại thông tin thêm. Trọng số Akaike đã chuyển đổi sự khác nhau thời gian thành xác suất về mô hình/con ngựa tốt nhất ở data/cuộc đua tương lai. Nhưng nếu điều kiện đường đua thay đổi, nhưng xác suất này sẽ gây nhầm. Dự đoán tương lai dựa vào một cuộc đua hay một lần fit là không bao giờ đảm bảo.</p>
    <p>Hãy nghĩ mô hình như những viên đá được ném nhảu trên mặt nước. Không viên đá nào sẽ tới được bờ bên kia (dự đoán hoàn hảo), nhưng vài viên đá sẽ đi xa hơn viên khác, theo trung bình (dự đoán data kiểm tra tốt hơn). Nhưng với mỗi lần ném, rất nhiều điều kiện độc nhất phải đạt được - cơn gió xuất hiện hay đổi hướng, con vịt xuất hiện chặn viên đá, hoặc người ném bị trượt tay. Cho nên viên nào đi xa nhất là không chắc chắn. Tuy nhiên, khoảng cách tương đối để đến bờ của mỗi viên sẽ cho thông tin viên nào là tốt nhất theo trung bình. Nhưng chúng ta không nên quá tự tin vào bất kỳ viên đá nào, trừ phi khoảng cách giữa những viên đá là rất lớn.</p>
    <p>Dĩ nhiên không minh hoạ ẩn dụ nào là hoàn hảo. Ẩn dụ luôn luôn vậy. Nhưng nhiều người sẽ thấy chúng có ích khi diễn giải tiêu chuẩn thông tin.</p>
</div>

### 7.5.2 Giá trị ngoại lai (outlier) và ảo tưởng khác

Trong ví dụ ly hôn ở Chương 5, chúng ta thấy dự đoán posterior có vài bang rất khó để mô hình dự đoán ngược.Cụ thể, bang Idaho giống như một **GIÁ TRỊ NGOẠI LAI (OUTLIER)**. Dữ liệu giống Idaho có khuynh hướng ảnh hưởng rất lớn đến hồi quy thông thường. Hãy xem PSIS và WAIC thể hiện mức độ quan trọng đó. Ta sẽ fit lại ba mô hình ở Chương 5.

```python
WaffleDivorce = pd.read_csv("https://github.com/rmcelreath/rethinking/blob/master/data/WaffleDivorce.csv?raw=true", sep=";")
d = WaffleDivorce
d["A"] = d.MedianAgeMarriage.pipe(lambda x: (x - x.mean()) / x.std())
d["D"] = d.Divorce.pipe(lambda x: (x - x.mean()) / x.std())
d["M"] = d.Marriage.pipe(lambda x: (x - x.mean()) / x.std())


def model(A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bA * A)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


m5_1 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_1, optim.Adam(1), Trace_ELBO(), A=d.A.values, D=d.D.values)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p5_1 = svi.get_params(state)


def model(M, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bM * M
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


m5_2 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_2, optim.Adam(1), Trace_ELBO(), M=d.M.values, D=d.D.values)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p5_2 = svi.get_params(state)


def model(M, A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + bM * M + bA * A)
    numpyro.sample("D", dist.Normal(mu, sigma), obs=D)


m5_3 = AutoLaplaceApproximation(model)
svi = SVI(model, m5_3, optim.Adam(1), Trace_ELBO(), M=d.M.values, A=d.A.values, D=d.D.values)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p5_3 = svi.get_params(state)
```

Nhìn vào tổng kết posterior, cần nhắc lại là tỉ lệ kết hôn ($M$) có ít ảnh hưởng đến tỉ lệ ly dị ($D$), một khi độ tuổi khi kết hôn ($A$) thêm vào ở mô hình `m5.3`. Ta so sánh PSIS của những mô hình này:

```python
post = m5_1.sample_posterior(random.PRNGKey(24071847), p5_1, (1000,))
logprob = log_likelihood(m5_1.model, post, A=d.A.values, D=d.D.values)["D"]
az5_1 = az.from_dict(
    posterior={k: v[None, ...] for k, v in post.items()},
    log_likelihood={"D": logprob[None, ...]},
)
post = m5_2.sample_posterior(random.PRNGKey(24071847), p5_2, (1000,))
logprob = log_likelihood(m5_2.model, post, M=d.M.values, D=d.D.values)["D"]
az5_2 = az.from_dict(
    posterior={k: v[None, ...] for k, v in post.items()},
    log_likelihood={"D": logprob[None, ...]},
)
post = m5_3.sample_posterior(random.PRNGKey(24071847), p5_3, (1000,))
logprob = log_likelihood(m5_3.model, post, A=d.A.values, M=d.M.values, D=d.D.values)[
    "D"
]
az5_3 = az.from_dict(
    posterior={k: v[None, ...] for k, v in post.items()},
    log_likelihood={"D": logprob[None, ...]},
)
az.compare({"m5.1": az5_1, "m5.2": az5_2, "m5.3": az5_3}, ic="loo", scale="deviance")
```

<p><samp><table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>loo</th>
      <th>p_loo</th>
      <th>d_loo</th>
      <th>weight</th>
      <th>se</th>
      <th>dse</th>
      <th>warning</th>
      <th>loo_scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m5.1</th>
      <td>0</td>
      <td>126.81</td>
      <td>4.25542</td>
      <td>0</td>
      <td>0.72456</td>
      <td>14.0231</td>
      <td>0</td>
      <td>False</td>
      <td>deviance</td>
    </tr>
    <tr>
      <th>m5.3</th>
      <td>1</td>
      <td>130.393</td>
      <td>6.36711</td>
      <td>3.583</td>
      <td>0.171027</td>
      <td>10.3578</td>
      <td>2.03122</td>
      <td>True</td>
      <td>deviance</td>
    </tr>
    <tr>
      <th>m5.2</th>
      <td>2</td>
      <td>139.885</td>
      <td>3.33723</td>
      <td>13.0747</td>
      <td>0.104413</td>
      <td>15.7063</td>
      <td>10.0529</td>
      <td>False</td>
      <td>deviance</td>
    </tr>
  </tbody>
</table></samp></p>

Có hai chuyện quan trọng ở đây. Đầu tiên mô hình không có tỉ lệ kết hôn, `m5.1`, trên đầu tiên. Nó là bởi vì tỉ lệ kết hôn có rất ít liên hệ với kết cục. Cho nên mô hình không có nó sẽ cho kết quả dự đoán ngoài mẫu tốt hơn một ít, ngay cả nó fit mẫu tệ hơn `m5.3`, mô hình chứa cả hai biến dự đoán. Sự khác nhau giữa 2 mô hình trên đầu chỉ có 2.49, với sai số chuẩn là 1.24, cho nên mô hình cho dự đoán cũng tương tự nhau. Đây là một minh hoạ điển hình, khi mà có vài biến có rất ít liên hệ với biến kết cục.

Thứ hai, ngoài bảng ở trên, bạn cũng nhận được một cảnh báo:

<samp>Estimated shape parameter of Pareto distribution is greater than 0.7 for one or more samples</samp>

Nó nghĩa là giá trị ước lượng làm mượt của PSIS không đáng tin cậy ở vài điểm. Nhớ lại trong phần PSIS, khi một mẫu quan sát có giá trị Pareto $k$ lớn hơn 0.5, trọng số quan trọng có thể không đáng tin cậy. Hơn nữa, những giá trị này có khuynh hướng là outlier với giá trị hiếm gặp, theo như mô hình. Kết quả là, những điểm này ảnh hưởng rất lớn và làm khó cho ước lượng độ chính xác dự đoán ngoài mẫu. Tại sao? Bởi vì mẫu mới sẽ hiếm khi có giá trị như những outlier, và bởi vì những outlier này ảnh hưởng lớn, chúng làm cho dự đoán ngoài mẫu tệ hơn mong đợi. WAIC cũng bị ảnh hưởng bởi outlier. Nó không cảnh báo tự động. Nhưng nó cũng có cách làm để định lượng nguy cơ này, thông quá ước lượng của số hạng phạt đền.

Nhìn lại từng bang, để thấy ai là người gây ra vấn đề này. Chúng ta có thể thêm `pointwise=True` cho PSIS. Khi đó, bạn sẽ có kết quả các giá trị $k$. Tôi cũng vẽ lên số hạng phạt đền từ WAIC, để cho thấy quan hệ giữa Pareto $k$ và số hạng phạt đền trong tiêu chuẩn thông tin.

![](/assets/images/fig 7-10.svg)
<details class="fig">
    <summary>Hình 7.10: Những mẫu quan sát có ảnh hướng lớn và dự đoán ngoài mẫu. Trục hoành là Pareto $k$ từ PSIS. Trục tung là số hạng phạt đền từ WAIC. Bang Idaho (ID) có giá trị hiếm gặp, theo như mô hình. Kết quả là nó có giá trị Pareto $k$ lớn và số hạng phạt đền trong WAIC lớn. Điểm như thế có ảnh hưởng lớn và giảm hiệu năng dự đoán.</summary>
    <pre><code>PSIS_m5_3 = az.loo(az5_3, pointwise=True, scale="deviance")
WAIC_m5_3 = az.waic(az5_3, pointwise=True, scale="deviance")
penalty = az5_3.log_likelihood.stack(sample=("chain", "draw")).var(dim="sample")
plt.figure(figsize=(3,3))
plt.plot(PSIS_m5_3.pareto_k.values, penalty.D.values, "o", mfc="none")
ax = plt.gca()
ax.set(xlabel="PSIS Pareto k", ylabel="WAIC penalty",
       xticks=[0,0.5,1],
       title="Gaussian model (m5.3)")
ax.vlines(0.5, 0, 2.5, linewidth=0.5, linestyle="dashed")
ax.annotate("ME", (0.3,0.8))
ax.annotate("ID", (1,2.3))</code></pre></details>

Biểu đồ này thể hiện Pareto $k$ và số hạng phạt đền cho từng mẫu quan sát. Bang Idaho (ID) có cả giá trị Pareto $k$ lớn (hơn 1) và số hạng phạt đền lớn (hơn 2). Như đã gặp ở Chương 5. Idaho có tỉ lệ ly dị thấp so với độ tuổi kết hôn. Kết quả là, nó ảnh hưởng rất lớn - nó cho nhiều ảnh hưởng đến phân phối posterior hơn những bang khác. Giá trị Pareto $k$ gấp đôi giá trị lý thuyết, ở đó phương sai trở nên vô hạn (đường nét đứt). Tương tự, WAIC gán Idaho số hạng phạt đền trên 2. Số hạng phạt đền này đôi khi được gọi là "số lượng parameter hiệu quả", bởi vì trong hồi quy tuyến tính thông thường, tổng của tất cả số hạng phạt đền từ mọi mẫu quan sát có khuynh hướng gần bằng số lượng parameter tự do trong mô hình. Nhưng trong trường hợp này có 4 parameter và tổng phạt đền gần bằng 6. Outlier Idaho là nguyên nhân gây ra nguy cơ overfitting này.

Vậy chúng ta làm được gì? Có một truyền thống loại bỏ outlier. Người ta thường bỏ outlier trước khi mô hình được fit, đơn thuần dựa trên độ lệch chuẩn từ trung bình kết cục. Bạn đừng nên làm như vậy - một mẫu quan sát chỉ có thể bị ngoài mong đợi và ảnh hưởng nhiều dưới một mô hình đã fit. Sau khi fit, bức tranh sẽ thay đổi. Nếu như có outlier, bạn cũng nên báo cáo kết quả khi có và không có chúng, việc loại bỏ outlier khi đó sẽ hợp lý hơn. Nhưng nếu có rất nhiều outlier và chúng cần phải mô hình chúng, thì làm sao?

Một vấn đề cơ bản của mô hình sai số Gaussian là nó rất dễ bị ngạc nhiên. Phân phối Gaussian có đuôi rất mỏng. Điều này nghĩa là có rất ít mật độ xác suất cho giá trị xa trung bình. Nhiều hiện tượng tự nhiên có đuôi mỏng như thế. Chiều cao con người là một ví dụ điển hình. Nhưng cũng có nhiều hiện tượng thì không. Thay vào đó nhiều hiện tượng có đuôi dày hơn với những quan sát hiếm gặp, cực trị. Chúng không phải do sai sót đo lường, nhưng là sự kiện thực chứa thông tin về mô hình xử lý tự nhiên.

Một cách khác để sử dụng những quan sát cực trị này và giảm thiểu ảnh hưởng của chúng là sử dụng **HỒI QUY ROBUST**. "Hồi quy robust" có rất nhiều nghĩa, những nó thường được hiểu là mô hình tuyến tính mà trong đó ảnh hưởng của quan sát cực trị bị giảm. Một loại hồi quy robust thông dụng và hữu ích là thay mô hình Gaussian với phân phối đuôi dày hơn như phân phối **STUDENT'S T** (hay "Student-t").<sup><a name="r131" href="#131">131</a></sup> Phân phối này không có liên quan đến học sinh (student). Phân phối Student-t xuất phát từ việc kết hợp phân phối Gaussian với nhiều phương sai khác nhau.<sup><a name="r132" href="#132">132</a></sup> Nếu phương sai được đa dạng hoá, thì đuôi có thể khá dày.

Phân phối Student-t tổng quát có parameter trung bình $\mu$ và độ lệch chuẩn (hay scale) $\sigma$ giống như Gaussian, nhưng nó cũng có thêm parameter hình dạng (shape) $\nu$ kiểm soát mức độ dày của đuôi. Khi $\nu$ lớn, đuôi dày lên, hội tụ đến giới hạn $\nu=\infty$ thành phân phối Gaussian. Nhưng khi $\nu$ đi tới 1, đuôi sẽ dày lên và quan sát hiếm gặp, cực trị sẽ xuất hiện thường xuyên hơn. Hình 7.11 so sánh phân phối Gaussian (màu xanh) tương ứng với phân phối Student-t (màu đỏ) với $\nu=2$. Phân phối Student-t có đuôi dày hơn, và rõ ràng hơn nếu ta thể hiện nó ở cân logarith, (bên phải), trong đó Gaussian có đuôi mỏng đi rất nhanh theo bậc 2 - phân phối bình thường là số $e$ luỹ thừa parabola - trong khi Student-t có đuôi mỏng đi chậm hơn.

![](/assets/images/fig 7-11.svg)
<details class="fig"><<summary>Hình 7.11: Những cái đuôi và mẫu quan sát ảnh hưởng lớn. Phân phối Gaussian màu xanh gán rất ít xác suất cho quan sát cực trị. Nó có đuôi mỏng. Phân phối Student-t (màu đỏ) có $\nu=2$ gán nhiều xác suất hơn cho quan sát cực trị. Phân phối được so sánh trên cân xác suất và cân logarith xác suất.</summary>
<pre><code>seq = jnp.linspace(-4,4)
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(seq, jnp.exp(dist.Normal().log_prob(seq)), label="Gaussian")
ax[0].plot(seq, jnp.exp(dist.StudentT(df=2).log_prob(seq)), label="Student-t")
ax[0].legend()
ax[0].set(xlabel='value', ylabel='density')
ax[1].plot(seq, -dist.Normal().log_prob(seq), label="Gaussian")
ax[1].plot(seq, -dist.StudentT(df=2).log_prob(seq), label="Student-t")
ax[1].legend()
ax[1].set(xlabel='value', ylabel='minus log density')</code></pre></details>

Nếu bạn có rất nhiều data có sự kiện như vậy, bạn có thể ước lượng $\nu$. Chuỗi thời gian trong kinh tế học, xảy ra trong một khoảng thời gian dài, là một ví dụ. Nhưng khi dùng hồi quy robust, chúng ta thường không cố gắng ước lượng $\nu$, bởi vì không có đủ quan sát cực trị để làm thế. Thay vào đó là giả định $\nu$ rất nhỏ (đuôi dày) để giảm thiểu ảnh hưởng của outlier. Ví dụ, nếu chúng ta dùng mức độ tàn phá của chiến tranh từ năm 1950 để ước lượng một khuynh hướng, ước lượng này sẽ bị nhiễu do những chiến tranh lớn như Thế Chiến I và Thế Chiến II rất hiếm. Chúng nằm ở phần đuôi dày của mức độ tàn phá.<sup><a name="r133" href="#133">133</a></sup> Một ước lượng hợp lý cần dựa trên chuỗi thời gian dài hơn hoặc sử dụng phân phối có đuôi dày.

Hãy thử ước lượng lại mô hình ly dị bằng phân phối Student-t có $\nu=2 $.

```python
def model(M, A, D=None):
    a = numpyro.sample("a", dist.Normal(0, 0.2))
    bM = numpyro.sample("bM", dist.Normal(0, 0.5))
    bA = numpyro.sample("bA", dist.Normal(0, 0.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + bM * M + bA * A
    numpyro.sample("D", dist.StudentT(2, mu, sigma), obs=D)


m5_3t = AutoLaplaceApproximation(model)
svi = SVI(
    model, m5_3t, optim.Adam(0.3), Trace_ELBO(), M=d.M.values, A=d.A.values, D=d.D.values
)
init_state = svi.init(random.PRNGKey(0))
state, loss = lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(1000))
p5_3t = svi.get_params(state)
```

Khi bạn tính PSIS, bạn sẽ không có cảnh báo về Pareto $k$ nữa. Ảnh hưởng ương đối của Idaho đã giảm. Nó ảnh hưởng thế nào đến phân phối posterior của quan hệ giữa tuổi kết hôn và tỉ lệ ly hôn? Nếu bạn so sánh `m5.3t` và `m5.3`, bạn sẽ thấy hệ số `bA` xa zero nhiều hơn khi bạn giới thiệt phân phối Student-t. Đó là bởi vì Idaho có tỉ lệ ly hôn thấp và độ tuổi kết hôn thấp. Mặc dù nó ảnh hưởng nhiều, nó giảm đáng kể quan hệ giữa tuổi kết hôn và ly hôn. Bây giờ nó ít ảnh hưởng hơn, cho nên hệ số ước lượng quan hệ lớn hơn. Nhưng hệ quả của việc dùng hồi quy robust không bao giờ là tăng thêm mối quan hệ. Nó dựa vào từng chi tiết.

Một thứ nữa là phân phối đuôi dày cho phép kiểm soát bất đồng giữa prior và data. Chúng ta sẽ quay lại vấn đề này ở chương sau, khi bạn bắt đầu dùng chuỗi Markov và suy ra được phân phối posterior không phải Gaussian.

<div class="alert alert-info">
    <p><strong>Lời nguyền của Tippecanoe.</strong> Một mối lo lắng trong so sánh mô hình là, nếu chúng ta cố gắng thử các kiểu kết hợp biến và chuyển đổi biến, thì một lúc nào đó chúng ta sẽ tìm ra một mô hình có thể fit bất kỳ mẫu nào cũng tốt. Nhưng mô hình sẽ bị overfit rất nhiều, không thể tổng quát hoá được. Và WAIC và những tiêu chuẩn tương tự sẽ bị lừa. Nó giống như câu chuyện <i>Lời nguyền Tippecanoe.</i><sup><a name="r134" href="#134">134</a></sup> Từ năm 1840 đến năm 1960, tất cẩ tổng thống Mỹ được trúng cử vào năm kết thúc bằng số 0 (xảy ra mỗi 20 năm) đã chết trong văn phòng. John F.Kennedy là người cuối cùng, trúng cử vào 1960 và bị ám sát vào năm 1963. Bảy tổng thống Mỹ chết lần lượt theo kiểu này. Ronald Reagan trúng cử vào 1980, nhưng mặc dù có ít nhất một lần suýt bị ám sát, ông ta đã sống lâu hơn, phá bỏ lời nguyền. Với đủ thời gian và dữ liệu, quy luật như vậy có thể được tìm thấy cho mọi loại data. Nếu chúng ta cố gắng tìm kiếm, chúng ta sẽ có nguy cơ dính lời nguyền của Tippecanoe.</p>
    <p>Đùa nghịch và chế tạo nhiều biến dự đoán là một cách tốt tìm ra sự trùng hợp ngẫu nhiên, nhưng không nhất thiết là cách tốt để đánh giá giả thuyết. Tuy nhiên, việc fit nhiều mô hình không phải luôn luôn là ý tưởng nguy hiểm, cho rằng đã đầu tư suy nghĩ về danh sách các biến số vào lúc đầu. Có hai trường hợp mà chiến thuật này được sử dụng. Một là, đôi khi chúng ta chỉ muốn khám phá một tập dữ liệu, bởi vì chưa có giả thuyết cụ thể để đánh giá. Đây chính là <strong>ĐẦY ĐOẠ DỮ LIỆU (DATA DREDGING)</strong>, mà người ta không chịu tự nhận. Nhưng khi sử dụng chung với trung bình hoá mô hình, và có thể chấp nhận rằng, nó là một cách để kích thích khám phá thêm về dữ liệu trong tương lai. Thứ hai, đôi khi chúng ta cần thuyết phục khán giá rằng chúng ta đã thử mọi cách kết hợp của các biến dự đoán, bởi vì không biến nào giúp ích được cho dự đoán.
    </p>
</div>

## <center>7.6 Tổng kết</center><a name="a6"></a>

Chương này là một chuyến đi dài. Nó bắt đầu bằng vấn đề overfitting, một hiện tượng chung khi mô hình nhiều parameter fit mẫu tốt hơn, ngay cả khi parameter thêm vào là vô nghĩa. Hai công cụ phổ biến được giới thiệu để đánh giá overfitting là regularizing prior và ước lượng độ chính xác ngoài mẫu (WAIC và PSIS). Regularizing prior giảm overfitting lúc ước lượng, và WAIC và PSIS giúp đánh giá mức độ overfitting. Trong thực hành, hàm `compare` sẽ giúp ta phân tích các tập mô hình fit chung một data. Nếu bạn muốn suy luận nhân quả, thì những công cụ này sẽ gây hiểu nhầm cho người sử dụng. Cho nên mô hình phải được thiết kế dưới những phương phải khác, không phải được chọn dựa trên độ chính xác dự đoán ngoài mẫu. Nhưng ước lượng nhân quả vẫn có thể bị overfit. Cho nên bạn phải luôn luôn xem xét overfitting, đo lường nó bằng WAIC/PSIS và giảm thiểu nó bằng regularization.

---

<details class="endnotes"><summary>Endnotes</summary>
<ol start="98" class='endnotes'>
<li><a name="98" href="#r98">98. </a>De Revolutionibus, Book 1, Chapter 10.</li>
<li><a name="99" href="#r99">99. </a>See e.g. Akaike (1978), as well as discussion in Burnham and Anderson (2002).</li>
<li><a name="100" href="#r100">100. </a>When priors are flat and models are simple, this will always be true. But later in the book, you’ll work with other types of models, like multilevel regressions, for which adding parameters does not necessarily lead to better fit to sample.</li>
<li><a name="101" href="#r101">101. </a>Data from Table 1 of McHenry and Coffing (2000).</li>
<li><a name="102" href="#r102">102. </a>Gauss 1809, Theoria motus corporum coelestium in sectionibus conicis solem ambientum.</li>
<li><a name="103" href="#r103">103. </a>See Grünwald (2007) for a book-length treatment of these ideas.</li>
<li><a name="104" href="#r104">104. </a>There are many discussions of bias and variance in the literature, some much more mathematical than others. For a broad treatment, I recommend Chapter 7 of Hastie, Tibshirani and Friedman’s 2009 book, which explores BIC, AIC, cross-validation and other measures, all in the context of the bias-variance trade-off.</li>
<li><a name="105" href="#r105">105. </a>I first encountered this kind of example in Jaynes (1976), page 246. Jaynes himself credits G. David Forney’s 1972 information theory course notes. Forney is an important figure in information theory, having won several awards for his contributions.</li>
<li><a name="106" href="#r106">106. </a>As of 2019, calibration and Brier scores are available online https://projects.fivethirtyeight.com/checkingour-work/. Silver (2012) contains a chapter, Chapter 4, that unfortunately pushes calibration as the most important diagnostic for prediction. There is a more nuanced endnote, however, that makes the same point as I do in the Rethinking box.</li>
<li><a name="107" href="#r107">107. </a>Calibration makes sense to frequentists, who define probability as objective frequency. Among Bayesians, in contrast, there is no agreement. Strictly speaking, there are no “true” probabilities of events, because probability is epistemological and nature is deterministic. See Jaynes (2003), Chapter 9. Gneiting et al. (2007) provide a flexible definition: Consistency between the distributional forecasts and the observations. They develop a useful approach, but they admit it has a “frequentist flavour” (page 264). No one recommends claiming that predictions are good, just because they are calibrated.</li>
<li><a name="108" href="#r108">108. </a>Shannon (1948). For a more accessible introduction, see the venerable textbook Elements of Information Theory, by Cover and Thomas. Slightly more advanced, but having lots of added value, is Jaynes’ (2003, Chapter 11) presentation. A foundational book in applying information theory to statistical inference is Kullback (1959), but it’s not easy reading.</li>
<li><a name="109" href="#r109">109. </a>See two famous editorials on the topic: Shannon (1956) and Elias (1958). Elias’ editorial is a clever work of satire and remains as current today as it was in 1958. Both of these one-page editorials are readily available online.</li>
<li><a name="110" href="#r110">110. </a>I really wish I could say there is an accessible introduction to maximum entropy, at the level of most natural and social scientists’ math training. If there is, I haven’t found it yet. Jaynes (2003) is an essential source, but if your integral calculus is rusty, progress will be very slow. Better might be Steven Frank’s papers (2009; 2011) that explain the approach and relate it to common distributions in nature. You can mainly hum over the maths in these and still get the major concepts. See also Harte (2011), for a textbook presentation of applications in ecology.</li>
<li><a name="111" href="#r111">111. </a>Kullback and Leibler (1951). Note however that Kullback and Leibler did not name this measure after themselves. See Kullback (1987) for Solomon Kullback’s reflections on the nomenclature. For what it’s worth, Kullback and Leibler make it clear in their 1951 paper that Harold Jeffreys had used this measure already in the development of Bayesian statistics.</li>
<li><a name="112" href="#r112">112. </a>In non-Bayesian statistics, under somewhat general conditions, a difference between two deviances has a chi-squared distribution. The factor of 2 is there to scale it the proper way. Wilks (1938) is the usually primordial citation.</li>
<li><a name="113" href="#r113">113. </a>See Zhang and Yang (2015).</li>
<li><a name="114" href="#r114">114. </a>Gelfand (1996).</li>
<li><a name="115" href="#r115">115. </a>Vehtari et al. (2016).</li>
<li><a name="116" href="#r116">116. </a>See Gelfand (1996). There is also a very clear presentation in Magnusson et al. (2019).</li>
<li><a name="117" href="#r117">117. </a>See Vehtari et al. (2019b).</li>
<li><a name="118" href="#r118">118. </a>Akaike (1973). See also Akaike (1974, 1978, 1981a), where AIC was further developed and related to Bayesian approaches. Ecologists tend to know about AIC from Burnham and Anderson (2002).</li>
<li><a name="119" href="#r119">119. </a>A common approximation in the case of small N is AICc = D train + 1−(k+1)/N 2k . As N grows, this expression approaches AIC. See Burnham and Anderson (2002).</li>
<li><a name="120" href="#r120">120. </a>Lunn et al. (2013) contains a fairly understandable presentation of DIC, including a number of different ways to compute it.</li>
<li><a name="121" href="#r121">121. </a>Quote in Akaike (1981b).</li>
<li><a name="122" href="#r122">122. </a>Watanabe (2010). Gelman et al. (2014) re-dub WAIC the “Watanabe-Akaike Information Criterion” to give explicit credit to Watanabe, in the same way people renamed AIC after Akaike. Gelman et al. (2014) is worthwhile also for the broad perspective it takes on the inference problem.</li>
<li><a name="123" href="#r123">123. </a>There was a tribal exchange over this issue in 2018. See Gronau and Wagenmakers (2019) and Vehtari et al. (2019c). The exchange focused on comparing Bayes factors to PSIS, but it is relevant to WAIC as well. This exchange is reminiscent of similar debates over BIC and AIC from the 1990s.</li>
<li><a name="124" href="#r124">124. </a>Schwarz (1978).</li>
<li><a name="125" href="#r125">125. </a>Gelman and Rubin (1995). See also section 7.4, page 182, of Gelman et al. (2013).</li>
<li><a name="126" href="#r126">126. </a>See Watanabe (2018b) and Watanabe (2018a). Watanabe has some useful material on his website. See http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/psiscv.html.</li>
<li><a name="127" href="#r127">127. </a>See results reported in Watanabe (2018b). See also Vehtari et al. (2016). See also some simulations reported on Watanabe’s website: http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/</li>
<li><a name="128" href="#r128">128. </a>This is closely related to minimum description length. See Grünwald (2007).</li>
<li><a name="129" href="#r129">129. </a>Aki Vehtari and colleagues are working on conditions for the reliability of the normal error approximation. It’s worth checking his working papers for updates.</li>
<li><a name="130" href="#r130">130. </a>The first edition had a section on model averaging, but the topic has been dropped in this edition to save space. The approach is really focused on prediction, not inference, and so it doesn’t fit the flow of the second edition. But it is an important approach. The traditional approach is to use weights to average predictions (not parameters) of each model. But if the set of models isn’t carefully chosen, one can do better with model “stacking.” See Yao et al. (2018).</li>
<li><a name="131" href="#r131">131. </a>The distributions name comes from a 1908 paper by William Sealy Gosset, which he published under the pseudonym “Student.” One story told is that Gosset was required by his employer (Guinness Brewery) to publish anonymously, or rather he voluntarily hid his identity, to disguise that Guinness was using statistics to improve beer. Regardless, the distribution was derived earlier in 1876, within the Bayesian framework. See Pfanzagl and Sheynin (1996).</li>
<li><a name="132" href="#r132">132. </a>Specifically, if the variance has an inverse-gamma distribution σ 2 ∼ inverse-gamma(ν/2, ν/2), then the resulting distribution is Student-t with shape parameter (degrees of freedom) ν.</li>
<li><a name="133" href="#r133">133. </a>See “The Decline of Violent Conflicts: What Do The Data Really Say?” by Pasquale Cirillo and Nassim Nicholas Taleb, Nobel Foundation Symposium 161: The Causes of Peace. You can find it readily by searching online.</li>
<li><a name="134" href="#r134">134. </a>William Henry Harrison’s military history earned him the nickname “Old Tippecanoe.” Tippecanoe was the sight of a large battle between Native Americans and Harrison, in 1811. In popular imagination, Harrison was cursed by the Native Americans in the aftermath of the battle.</li>
</ol>
</details>