---
title: "Chapter 8: Conditional Manatees"
description: "Chương 8: Những con lợn biển có điều kiện"
---


- [8.1 Xây dựng sự tương tác](#a1)
- [8.2 Tính đối xứng của tương tác](#a2)
- [8.3 Tương tác liên tục](#a3)
- [8.4 Tổng kết](#a4)

<details class='imp'><summary>import lib cần thiết</summary>
{% highlight python %}import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO, log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation
az.style.use("fivethirtyeight"){% endhighlight %}</details>

Lợn biển (manatee - *Trichechus manatus*) là một động vật hữu nhũ chậm chạp, sống dưới vùng nước ấm và nông. Lợn biển không có thiên dịch tự nhiên, nhưng chúng có chia sẽ nước với những chiếc thuyền máy. Và thuyền máu có cánh quạt. Khi những con lợn biển liên quan với loài voi và có lớp da rất dày, cánh quạt có thể giết chúng. Phần lớn các con lợn biển trưởng thành đều có sẹo từ va chạm với thuyền ([**HÌNH 8.1, trên**](#f1))

<a name="f1"></a>![](/assets/images/fig 8-1.png)
<details class="fig"><summary>Hình 8.1: TRÊN: Các vết sẹo trên lưng của 5 con lợn biển Florida. Các dãy trầy xước, như trên các cá thể Africa và Flash, là dấu hiệu của vết thương do cánh quạt. DƯỚI: Ba ví dụ thiệt hại trên những chiếc máy bay thả bom A.W.38 sau nhiệm vụ.</summary></details>

Armstrong Whitworth A.W.38 Whitley là máy bay thả bom tiền tuyến của Lực Lượng Hàng Không Hoàng Gia. Vào Thế Chiến II, A.W.38 mang bom và thả vào địa phận của Đức. Không giống như lợn biển, A.W.38 có thiên địch tự nhiên hung tợn: pháo phản không. Nhiều máy bay không bao giờ trở về từ nhiệm vụ. Và thiệt hại trên những chiếc còn sống sót đã chứng minh điều đó ([**HÌNH 8.1, dưới**](#f1))

Lợn biển có giống máy bay A.W.38 không? Trong cả hai trường hợp - vết trầy do máy quạt của lợn biển và các lỗ thủng do đạn của máy bay - chúng ta muốn làm gì đó để cải thiện khả năng giúp lợn biển và máy bay sống sót. Nhiều người quan sát nghĩ rằng giúp lợn biển và máy bay nghĩa là giảm loại tổn thương mà chúng ta nhìn thấy trên chúng. Với lợn biển, nó nghĩa là cần thêm lớp bảo vệ trên cánh quạt (của thuyền, không phải của lợn biển). Với máy bay, nó nghĩa là cần thêm giáp ở những bộ phận máy bay có nhiều tổn thương nhất.

Trong cả hai trường hợp, bằng chứng đã gây hiểu sai. Cánh quạt không gây ra những tổn thương và cái chết cho lợn biển. Thực vậy tử thiết trên chúng khẳng định sự va chạm với các bộ phận tù của chiếc thuyền nhưng mái chéo, cho tổn thương nhiều hơn. Tương tự, tăng giáp ở những thành phần tổn thương không cải thiện cho các máy bay thả bom. Thay vào đó, nâng cấp máy bay thả bom A.W.38 nghĩa là nâng giáp các bộ phận *không bị tổn thương*. Ví dụ từ những con lợn biển và máy bay thả bom còn sống sót là gây hiểu sai, bởi vì nó *đặt điều kiện* trên sống còn. Lợn biển và máy bay bị tiêu diệt trông sẽ khác. Một con lợn biển bị va chạm với máu chèo sẽ ít khả năng sống sót hơn nhưng con bị cánh quạt trợt qua. Nên giữa các con sống sót, vết trầy do cánh quạt là thường gặp. Tương tự, những máy bay trở về được không có vết thương rõ ràng trên động cơ và buồng lái. Họ may mắn. Máy bay thả bom không về nhà được thì ít như vậy hơn. Để có được đáp án chính xác, trong hai trường hợp trên, chúng ta cần phải nhận ra loại tổn thương nhìn thấy này là được đặt điều kiện trên sống còn.

**ĐẶT ĐIỀU KIỆN** là một trong những nguyên tắc quan trọng nhất trong suy luận thống kê. Data, như các vết xước lợn biển và tổn thương máy bay, được đặt điều kiện trên cách chúng được đưa vào mẫu. Phân phối posterior được đặt điều kiện trên data. Tất cả mọi suy luận bằng mô hình được đặt điều kiện trên mô hình. Mọi suy luận được đặt điều kiện trên một thứ gì đó, cho dù chúng ta có nhận ra hay không.

Và một phần lớn sức mạnh của mô hình thống kê đến từ việc tạo ra các thiết bị cho phép xác suất được đặt điều kiện trên các khía cạnh của từng trường hợp. Mô hình tuyến tính mà bạn quen thuộc mà thiết bị thô sơ cho phép mọi kết cục $y_i$ được đặt điều kiện trên một tập các biến dự đoán cho mỗi trường hợp $i$. Giống như các epicycle của mô hình Ptolemaic và Kopernikan (Chương 4 và 7), mô hình tuyến tính cho chúng ta phương pháp mô tả khả năng đặt điều kiện.

Mô hình tuyến tính đơn giản thông thường không có khả năng cung cấp đủ các phép đặt điều kiện, tuy nhiên . Mọi mô hình đến bây giờ trong sách này giả định mỗi biến dự đoán có mối quan hệ độc lập với trung bình của kết cục. Nhưng nếu chúng ta muốn cho phép mối quan hệ được đặt điều kiện thì sao? Ví dụ, trong data sữa các loài khỉ từ các chương trước, giả sử quan hệ giữa năng lượng sữa và kích thước não thay đổi theo nhóm loài (khỉ, vượn, tinh tinh). Điều này giống như nói rằng ảnh hưởng của kích thước não trên năng lượng sữa được đặt điều kiện trên nhóm loài. Mô hình tuyến tính từ chương trước không giải quyết được câu hỏi này.

Để mô hình hoá điều kiện sâu hơn - khi mức độ quan trọng của một biến phụ thuộc vào một biến dự đoán khác - chúng ta cần **SỰ TƯƠNG TÁC (INTERACTION)** (cũng được biết **SỰ ĐIỀU TIẾT - MODERATION**). Tương tác là một loại đặt điều kiện, một cách để cho phép tham số (thực ra là phân phối posterior) được đặt điều kiện trên các khía cạnh xa hơn của data. Loại tương tác đơn giản nhất, tương tác tuyến tính, được xây dựng bằng cách mở rộng chiến thuật mô hình tuyến tính vào tham số trong mô hình tuyến tính. Cho nên nó đồng nghĩa với việc thay thế epicycle trên epicycle trong mô hình Ptolemaic và Kopernikan. Nó mang tính mô tả, nhưng rất mạnh.

Tổng quát hơn, tương tác là trung tâm của đa số mô hình thống kê đằng sau thế giới ấm áp của kết cục Gaussian và mô hình tuyến tính của trung bình. Trong mô hình tuyến tính tổng quát (GLM, Chương 10 và sau đó), ngay cả khi người ta không định nghĩa rõ ràng các biến là tương tác, chúng vẫn tương tác ở một mức độ nào đó. Mô hình đa tầng cũng cho hiệu ứng tương tự. Mô hình đa tầng thông thường là một mô hình tương tác khổng lồ, trong đó các giá trị ước lượng (intercept và slope) được đặt điều kiện cho cụm (người, loài, làng, thành phố, vũ trụ) trong data. Hiệu ứng tương tác đa tầng là phức tạp. Chúng không chỉ cho phép ảnh hưởng của biến dự đoán thay đổi phụ thuộc vào một biến khác, mà còn ước lượng khía cạnh của *phân phối* của những thay đổi đó. Điều này nghe có vẻ thiên tài, hoặc điên rồ, hoặc cả hai. Cho dù thế nào, bạn không thể có sức mạnh của mô hình tầng mà không có nó.

Mô hình cho phép tương tác phức tạp thì dễ fit vào data. Nhưng chúng cũng được cho là khó hiểu hơn. Và nên tôi dành chương này nói về các hiệu ứng tương tác đơn giản: làm sao để định nghĩa, diễn giải, và minh hoạ chúng. Chương này bắt đầu bằng một trường hợp tương tác giữa một biến phân nhóm và một biến liên tục. Trong bối cảnh này, rất dễ để nhận ra dạng giả thuyết cho phép sự tương tác. Rồi sau đó chương này nói về tương tác phức tạp hơn giữa các biến dự đoán liên tục. Nó khó hơn. Trong tất cả các phần của chương này, dự đoán của mô hình được minh hoạ, trung bình hoá trên tính bất định trong tham số.

Sự tương tác là bình thường, nhưng chúng không dễ. Hi vọng là chương này là tạo một nền tảng vững chắc cho việc diễn giải mô hình tuyến tính tổng quát và mô hình đa tầng trong các chương sau.

<div class="alert alert-info">
<p><strong>Minh tinh thống kê, Abraham Wald.</strong> Câu chuyện máy bay thả bom trong Thế Chiến II là tác phẩm của Abraham Wald (1902-1950). Wald sinh ra ở nơi mà bây giờ gọi là Romania, nhưng di cư sang Mỹ sau khi Nazi xâm chiếm nước Áo. Wald đã cống hiến rất nhiều trong cuộc đời ngắn ngủi của ông. Có lẽ công tình liên quan nhất đến tài liệu này, là Wald đã chứng mình rằng với nhiều loại quy luật để quyết định theo thống kê, luôn luôn tồn tại một quy luật Bayes chí ít tốt bằng nhiều quy luật non-Bayes. Wald đã chứng minh điều này, một cách xuất sắc, bắt đầu với các tiền đề non-Bayes, và nên phe anti-Bayes không thể mặc kệ nó nữa. Công trình này được tóm tắc trong sách 1950 của Wald, được phát hành chỉ trước khi ông mất. Wald chết khi quá trẻ, từ một vụ rơi máy bay khi tham quan Ấn Độ.</p></div>

## <center></center>8.1 Xây dựng sự tương tác<a name="a1"></a>

Châu Phi rất đặc biệt. Lục địa lớn thứ hai, đa dạng về văn hoá và di truyền. Châu Phi có 3 tỉ người ít hơn so với Châu Á, nhưng nó lại có nhiều ngôn ngữ giao tiếp. Châu Phi đa dạng di truyền và đa số các biến thể di truyền ngoài Châu Phi là một phần nhỏ của biến thể trong Châu Phi. Châu Phi cũng đặc biệt về địa hình, theo một cách kỳ lạ: Địa hình xấu thường liên quan quan đến kinh tế xấu ngoài Châu Phi, nhưng kinh tế ở Châu Phi lại thực ra hưởng lợi từ địa hình xấu.

Để hiểu sự kỳ lạ này, hãy nhìn vào hồi quy của mức độ gồ ghề địa hình - một loại địa hình xấu - đối với hiệu năng kinh tế (log GDP trên đầu người vào năm 2000), cả trong và ngoài Châu Phi ([**HÌNH 8.2**](#f2)). Biến số `rugged` là Chỉ Số Gồ Ghề Địa Hình dùng để định lượng tính hỗn tạp cấu trúc của một vùng đất. Biến kết cục ở đây là logarith của tổng sản phẩm nội địa (gross domestic product - GDP) bình quân đầu người, từ năm 2000, `rgdppc_2000`. Chúng tôi sử dụng logarith của nó, bởi vì logarith của GDP là *mức độ* của GDP. Bởi vì sự giàu có tạo ra sự giàu có, nó có xu hướng tăng luỹ thừa liên quan với bất cứ thứ gì làm nó tăng. Nó giống như nói rằng khoảng cách tuyệt đối trong sự giàu có tăng ngày càng lớn, khi đất nước giàu có hơn. Cho nên khi chúng ta làm việc với logarith, chúng ta đang làm việc trên thang mức độ được chia đều hơn. Cho dù thế nào, hãy nhớ rằng chuyển đổi log không làm mất thông tin. Nó chỉ thay đổi những giả định của mô hình về hình dáng của quan hệ giữa các biến. Trong trường hợp này, GDP thô không quan hệ tuyến tính với bất cứ thứ gì, bởi vì hình dạng luỹ thừa của nó. Nhưng log GPD lại quan hệ tuyến tính với rất nhiều thứ.

<a name="f2"></a>![](/assets/images/fig 8-2.svg)
<details class="fig"><summary>Hình 8.2: Hồi quy tuyến tính khác nhau giữa trong và ngoài Châu Phi, giữa log GDP và mức độ gồ ghề địa hình. Slope dương trong Châu Phi, nhưng là âm khi ở ngoài. Làm sao chúng ta có thể tái hiện sự đảo ngược của slope, bằng data kết hợp?</summary>
{% highlight python %}url = r'https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv'
df = pd.read_csv(url+"?raw=true",sep=';')
df['rug_std']= df['rugged'].pipe(lambda x : (x-x.min())/(x.max()-x.min()))
df['log_gdp']= np.log(df['rgdppc_2000'])
df = df.dropna(subset=['log_gdp'])
df['log_gdp_p'] = df['log_gdp']/(df['log_gdp'].mean())
fig, axs = plt.subplots(1,2,figsize=(10,5))
sns.regplot(x=df['rug_std'][df['cont_africa']==1],
            y=df['log_gdp_p'][df['cont_africa']==1],ax=axs[0])
sns.regplot(x=df['rug_std'][df['cont_africa']==0],
            y=df['log_gdp_p'][df['cont_africa']==0],ax=axs[1])
for ax in axs:
    ax.set(xlabel="độ gồ ghề (chuẩn hoá)", ylabel="log GDP (tỉ lệ với trung bình)")
axs[0].set(title="Quốc gia Châu Phi")
axs[1].set(title="Quốc gia không Châu Phi")
for c in ['Lesotho', 'Seychelles']:
    entry = df[df['country']==c]
    axs[0].annotate(c, (entry['rug_std'],entry['log_gdp_p']))
for c in ['Switzerland', 'Tajikistan']:
    entry = df[df['country']==c]
    axs[1].annotate(c, (entry['rug_std'],entry['log_gdp_p'])){% endhighlight %}</details>

Chuyện gì đã xảy ra trong hình này? Đồng ý là độ gồ ghề liên quan với những nước nghèo, ở phần lớn thế giới. Địa hình gồ ghề nghĩa là giao thông khó khăn hơn. Tức là tiếp cận thị trường bị cản trở. Tức là giảm tổng sản phẩm nội địa. Cho nên quan hệ trái chiều trong Châu Phi là khó hiểu. Tại sao địa hình khó khăn lại liên quan đến GPD bình quân đầu người cao hơn?

Nếu quan hệ này là nhân quả, thì có thể bởi vì những vùng đất gồ ghề của Châu Phi được bảo vệ tránh khỏi những vụ mua bán nô lệ ở Đại Tây Dương và Ấn Độ Dương. Những người mua bán nô lệ thường nhắm vào những địa bàn dễ tấn công, với những đường đi dễ dàng ra biển. Những vùng này thường chịu đựng việc mua bán nô lệ tiếp tục bị thoái hoá kinh tế, sau khi thị trường nô lệ giảm đi. Và mức độ gồ ghề là tương quan với những đặc điểm địa hình khác, như đường bờ biển, nó cũng ảnh hưởng kinh tế. Cho nên rất khó để chắc chắn những gì đang xảy ra.

Giả thuyết nhân quả, dưới dạng DAG, có thể là (xem thêm cuối phần này):

![](/assets/images/dag 8-1.svg)

Trong đó $R$ là độ gồ ghề địa hình, $G$ là GDP, $C$ là lục địa, và $U$ là một tập nhiễu không quan sát được (như khoảng cách đến bờ biển). Hãy tạm mặc kệ $U$. Bạn sẽ xem xét những yếu tố nhiễu khác trong phần thực hành. Thay vào đó tập trung vào những gợi ý mà $R$ và $C$ đều ảnh hưởng $G$. Điều này có nghĩa là nguồn ảnh hưởng độc lập hoặc có thể chúng tương tác (một biến điều hoà ảnh hưởng của biến còn lại). DAG này không thể hiện sự tương tác. Bởi vì DAG không cụ thể cách mà hai biến kết hợp ảnh hưởng lên biến khác. DAG trên chỉ gợi ý rằng có một hàm sử dụng cả $R$ và $C$ để tạo ra $G$. Trong ký hiệu điển hình $G=f(R,C)$.

Cho nên chúng ta cần một cách tiếp cận thống kê để đánh giá những định đề khác nhau cho $f(R,C)$. Làm sao tạo mô hình cho khả năng đặt điều kiện trong [**HÌNH 8.2**](#f2)? Chúng ta có thể đánh lừa bằng cách chia data thành hai DataFrame, một cho Châu Phi và một cho những lục địa khác. Nhưng nó không phải ý tưởng tốt để chia data như vậy. Đây là bốn lý do.

Thứ nhất, thông thường có vài tham số, như $\sigma$, mà mô hình cho rằng không phụ thuộc vào lục địa nào. Bằng cách chia thành các bảng, bạn đã gây tổn hại đến độ chính xác của ước lượng cho những tham số này, bởi vì bạn đã làm cho ước lượng kém chính xác hai lần thay vì gồm tất cả những bằng chứng vào một trị số ước lượng. Do đó, bạn đã vô tình giả định rằng phương sai khác nhau giữa Châu Phi và các quốc giá không phải Châu Phi khác. Bây giờ, không có gì sai với giả định đó. Nhưng bạn muốn tránh những giả định vô ý.

Thứ hai, để đạt được mệnh đề xác suất của biến số mà bạn dùng để tách data ra, `cont_africa` trong trường hợp này, bạn cần thêm nó vào mô hình. Nếu không, bạn sẽ có một kết luận thống kê yếu. Có hay không tính bất định của một giá trị dự đoán phân biệt giữa Châu Phi và quốc gia không phải Châu Phi? Dĩ nhiên là có. Nếu bạn phân tích toàn bộ data trong một mô hình duy nhất, bạn sẽ không dễ định lượng tính bất định đó. Nếu bạn chỉ để phân phối posterior làm việc cho bạn, bạn sẽ có một ước lượng hữu ích về tính bất định đó.

Thứ ba, chúng ta muốn sử dụng tiêu chuẩn thông tin hoặc một phương thức khác để so sánh mô hình. Để so sánh một mô hình mà xem tất cả các lục địa theo cùng một cách với một mô hình cho phép slope khác nhau trong lục địa khác nhau, chúng ta cần mô hình sử dụng tất cả cùng một data (như đã giải thích trong Chương 7). Điều này có nghĩa chúng ta không thể tách data thành hai mô hình riêng biệt. Chúng ta cần phải để cho một mô hình duy nhất tách data nội tại.

Thứ tư, khi bạn bắt đầu sử dụng mô hình đa tầng (Chương 13), bạn sẽ thấy rằng có nhiều lợi ích thì mượn thông tin xuyên suốt phân nhóm như "Châu Phi" và "không Châu Phi". Điều này đúng đặc biệt khi cỡ mẫu thay đổi giữa các phân nhóm, và nguy cơ overfitting là cao hơn trong một vài nhóm. Nói cách khác, những gì chúng ta học về mức độ gồ ghề ngoài Châu Phi nên có vài hiệu ứng lên ước lượng trong Châu Phi, và ngược lại. Mô hình đa tầng (Chuonge 13) mượn thông tin bằng cách này, để cải thiện ước lượng cho mọi phân nhóm. Khi chúng ta tách data, việc mượn này là không khả thi.

<div class="alert alert-dark">
<p><strong>Nhân quả không dễ dàng.</strong> DAG mức đồ gồ ghề địa hình trong phần trước là dễ. Nhưng sự thật không đơn giản như vậy. Lục địa không phải nguồn quan tâm chính. Có thể có sự phơi nhiễm trong lịch sử theo lý thuyết của chủ nghĩa thực dân và mua bán nô lệ đã ảnh hưởng dai dẳng đến hiệu suất kinh tế. Những đặc tính của địa hình, như độ gồ ghề, theo nhân quả giảm những yếu tố lịch sử đó có thể ảnh hưởng gián tiếp vào kinh tế. Như vậy:</p>
<img src="./assets/images/dag 8-2.svg">
<p>$H$ là những yếu tố lịch sử như phơi nhiễm mua bán nô lệ. Tổng hiệu ứng nhân quả của $R$ bao gồm con đường trực tiếp $R \to G$ (được giả định là luôn luôn âm) và con đường gián tiếp $R \to H \to G$. Con đường thứ hai là con đường mà cùng biến thiên với lục địa $C$, bởi vì $H$ là quan hệ mạnh với $C$. Chú ý rằng biến nhiễu $U$ có thể ảnh hưởng bất kỳ biến nào (ngoại trừ $C$). Ví dụ nếu khoảng cách đến bờ biến là biến thực sự ảnh hưởng $H$ trong quá khứ, không phải độ gồ ghề địa hình, thì quan hệ giữa độ gồ ghề địa hình với GDP là không nhân quả. Data gồm một lượng lớn yếu tố nhiễu tiềm nằng mà bạn nên xem xét. Hệ thống tự nhiên như thế là phức tạp kinh khủng.</p></div>

### 8.1.1 Tạo mô hình gồ ghề

Hãy xem cách tái diễn sự đảo chiều slope này, chỉ trong một mô hình. Chúng ta bắt đầu bằng fit một mô hình duy nhất với mọi data, bỏ qua lục địa. Nó cho phép chúng ta suy nghĩ về cấu trúc mô hình và prior trước khi đối mặt với ác quỷ tương tác. Để bắt đầu, tải data về và tiền xử lý:

<b>code 8.1</b>
```python
d = pd.read_csv(r'https://github.com/rmcelreath/rethinking/blob/master/data/rugged.csv?raw=true', sep=";")
# make log version of outcome
d["log_gdp"] = d["rgdppc_2000"].apply(math.log)
# extract countries with GDP data
dd = d[d["rgdppc_2000"].notnull()].copy()
# rescale variables
dd["log_gdp_std"] = dd.log_gdp / dd.log_gdp.mean()
dd["rugged_std"] = dd.rugged / dd.rugged.max()
```

Mỗi dòng trong data là một quốc gia, và nhiều cột trong đó là kinh tế, địa hình và đặc tính lịch sử. GDP thô và độ gồ ghề địa hình không có ý nghĩa lắm cho con người. Cho tên tôi đã chuẩn hoá các biến để thành đơn vị dễ sử dụng hơn. Chuẩn hoá như thường lệ bằng cách trừ trung bình và chia cho độ lệch chuẩn. Nó giúp các biến trở thành z-score. Chúng ta không muốn thực hiện nó ở đây, bởi vì độ gồ ghề bằng không là có ý nghĩa. Cho nên thay vì độ gồ ghề được chia cho giá trị quan sát lớn nhất. Nghĩa là nó được chuẩn hoá thành thang đo từ hoàn toàn phẳng (zero) thành tối đa ở mẫu là 1 (Lesotho, một nơi rất gồ ghề và xinh đẹp). Tương tự, log GDP đuọc chia cho giá trị trung bình. Nên nó được chỉnh lại thành thang đo tỉ lệ với trung bình quốc tế. 1 là trung bình, 0.8 là 80% của trung bình, và 1.1 và 10% hơn trung bình

Để xây dựng mô hình Bayes cho quan hệ này, chúng ta lần nữa sử dụng một khung địa tâm:

$$\begin{matrix}
\log(y_i) &\sim \text{Normal}(\mu_i, \sigma)\\
\mu_i &= \alpha + \beta(r_i - \bar{r})\\
\end{matrix}$$

Trong đó, $y_i$ là GDP quốc gia $i$, $r_i$ là độ gồ ghề địa hình của quốc gia $i$, và $\bar{r}$ là độ gồ ghề trung bình của toàn bộ mẫu. Giá trị của nó là 0.215 - hầu hết các quốc gia là không gồ ghề. Nhớ rằng sử dụng $\bar{r}$ chỉ làm nó dễ hơn khi gán prior vào intercept $\alpha$.

Cái khó khăn ở đây là khi chúng ta cụ thể hoá prior. Nếu bạn giống tôi, bạn không có nhiều thông tin khoa học về quan hệ phù hợp giữa log GDP và độ gồ ghề địa hình. Nhưng ngay khi chúng ta không biết nhiều về bối cảnh, cách thức đo đạc tự nó ràng buộc prior theo những cách hữu ích. Kết cục và biến dự đoán được chuẩn hoá sẽ làm điều này dễ dàng hơn. Xem xét intercept trước, $\alpha$, định nghĩa là log GDP khi độ gồ ghề ở trung bình mẫu. Cho nên nó phải gần bằng 1, bởi vì chúng ta chuẩn hoá kết cục để trung bình là 1. Hãy bắt đầu bằng việc đoán:

$$ \alpha \sim \text{Normal}(0,1)$$

Bây giờ với slope $\beta$. Nếu chúng ta đặt nó ở giữa zero, tức là không có sai lệch dương hoặc âm, thì nó hợp lý. Những với độ lệch chuẩn thì sao? Hãy bắt đầu bằng đoán con số 1:

$$ \beta \sim \text{Normal}(0,1)$$

Chúng ta sẽ đánh giá ước đoán này bằng cách mô phỏng phân phối dự đoán prior. Điều cuối cùng chúng ta cần là prior của $\sigma$. Hãy gán nó bằng một thứ gì đó rộng, $\sigma \sim \text{Exponential}(1)$. Trong thực hành cuối chương, tôi sẽ yêu cầu bạn đối đầu với prior này. Nhưng chúng ta hãy bỏ nó qua trong ví dụ này.

Sau cùng, chúng ta có được mô hình ứng cử cho data gồ ghề địa hình:

<b>code 8.2</b>
```python
def model(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_1, losses = svi.run(random.PRNGKey(0), 1000)
```

Chúng ta sẽ chưa nhìn vào dự đoán posterior, mà vào dự đoán prior. Hãy trích xuất prior và thể hiện những đường thẳng suy ra từ nó.

<b>code 8.3</b>
```python
predictive = Predictive(m8_1.model, num_samples=1000, return_sites=["a", "b", "sigma"])
prior = predictive(random.PRNGKey(7), rugged_std=0)
# set up the plot dimensions
plt.subplot(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="ruggedness", ylabel="log GDP")
plt.gca().axhline(dd.log_gdp_std.min(), ls="--")
plt.gca().axhline(dd.log_gdp_std.max(), ls="--")
# draw 50 lines from the prior
rugged_seq = jnp.linspace(-0.1, 1.1, num=30)
mu = Predictive(m8_1.model, prior, return_sites=["mu"])(
    random.PRNGKey(7), rugged_std=rugged_seq
)["mu"]
for i in range(50):
    plt.plot(rugged_seq, mu[i], "k", alpha=0.3)
```

<a name="f3"></a>![](/assets/images/fig 8-3.svg)
<details class="fig"><summary>Hình 8.3: Mô phỏng để tìm kiếm prior hợp lý cho ví dụ độ gồ ghề địa hình. Đường nằm ngang chỉ điểm các quan sát GDP tối đa và tối thiểu quan sát được. Trái: lần đoán đầu tiên với prior mơ hồ. Phải: Mô hình cải tiến với những prior phù hợp hơn.</summary>
{% highlight python %}def modela(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 1))
    b = numpyro.sample("b", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
def modelb(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
fig, axs = plt.subplots(1,2, figsize=(10,5))
for ax, model in zip(axs, [modela, modelb]):
    guide = AutoLaplaceApproximation(model)
    svi = SVI(
        model, guide, optim.Adam(0.1), Trace_ELBO(),
        rugged_std=dd.rugged_std.values,
        log_gdp_std=dd.log_gdp_std.values,
    )
    params, losses = svi.run(random.PRNGKey(0), 1000)
    predictive = Predictive(model, num_samples=1000, return_sites=["a", "b", "sigma"])
    prior = predictive(random.PRNGKey(7), rugged_std=0)
    ax.set(xlim=(0, 1), ylim=(0.5, 1.5), xlabel="độ gồ ghề", ylabel="log GDP (tỉ lệ với trung bình)")
    ax.axhline(dd.log_gdp_std.min(), ls="--")
    ax.axhline(dd.log_gdp_std.max(), ls="--")
    rugged_seq = jnp.linspace(-0.1, 1.1, num=30)
    mu = Predictive(model, prior, return_sites=["mu"])(
        random.PRNGKey(7), rugged_std=rugged_seq
    )["mu"]
    for i in range(50):
        ax.plot(rugged_seq, mu[i], "C0", alpha=0.3)
axs[0].plot(rugged_seq, 1.3-.7*rugged_seq, "C1")
axs[0].set_title('a ~ Normal(1, 1)\nb ~ Normal(0, 1)')
axs[1].set_title('a ~ Normal(1, 0.1)\nb ~ Normal(0, 0.3)'){% endhighlight %}</details>

Kết quả được thể hiện ở bên trái [**HÌNH 8.3**](#f3). Đường nét đứt nằm ngang là tối đa và tối thiểu của giá trị log GDP quan sát. Những đường hồi quy có xu hướng cả dương và âm, như đáng lẽ nó vậy, nhưng đa số những đường này nằm trong vùng bất khả thi. Chỉ xem xét thang đo lường, những đường thẳng này nên đi qua gần điểm và độ gồ ghề là trung bình (0.125 ở trục hoành) và tỉ lệ log GDP là 1. Thay vào đó có nhiều đường mong đợi GDP trung bình nằm ngoài khoảng quan sát. Cho nên chúng ta cần prior $\alpha$ có độ lệch chuẩn hẹp hơn. $\alpha \sim \text{Normal}(0,0.1)$ sẽ đặt hầu hết tính phù hợp trong khoảng GDP quan sát được. Nhớ rằng: 95% của mật độ Gaussian là trong 2 độ lệch chuẩn. Cho nên prior Normal(0, 0.1) gán 95% tính phù hợp giữa 0.8 và 1.2. Nó vẫn còn mơ hồ, nhưng ít ra không phi lý.

Cùng lúc này, slope cũng rất biến thiên. Nó không phù hợp khi độ gồ ghề giải thích hầu như những biến thiên quan sát được trong log GDP. Một quan hệ mạnh không phù hợp, ví dụ, một đường thẳng đi từ độ gồ ghề tối thiểu và GDP cực ở một đầu đến độ gồ ghề tối đa và cực đối diện GDP ở đầu còn lại. tôi đã đánh dấu đường thẳng này bằng màu đỏ. Slope của đường này phải gần bằng 1.3 - 0.7 = 0.6, hiệu giữa tỉ lệ log GDP tối đa và tối thiểu quan sát được. Nhưng rất nhiều đường thẳng trong prior có slope mạnh hơn slope này. Dưới prior $\beta \sim \text{Normal}(0, 1)$, hơn một nửa slope sẽ có giá trị tuyệt đối lớn hơn 0.6.

<b>code 8.4</b>
```python
jnp.sum(jnp.abs(prior["b"]) > 0.6) / prior["b"].shape[0]
```
<samp>0.545</samp>

Hãy thay bằng $\beta \sim \text{Normal}(0, 0.3)$. Prior này có slope 0.6 là hai độ lệch chuẩn. Nó có một ít quá phù hợp, nhưng tốt hơn trước nhiều.

Với hai thay đổi này, mô hình bây giờ là:

<b>code 8.5</b>
```python
def model(rugged_std, log_gdp_std=None):
    a = numpyro.sample("a", dist.Normal(1, 0.1))
    b = numpyro.sample("b", dist.Normal(0, 0.3))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = numpyro.deterministic("mu", a + b * (rugged_std - 0.215))
    numpyro.sample("log_gdp_std", dist.Normal(mu, sigma), obs=log_gdp_std)
m8_1 = AutoLaplaceApproximation(model)
svi = SVI(
    model,
    m8_1,
    optim.Adam(0.1),
    Trace_ELBO(),
    rugged_std=dd.rugged_std.values,
    log_gdp_std=dd.log_gdp_std.values,
)
p8_1, losses = svi.run(random.PRNGKey(0), 1000)
```

Bạn có thể trích xuất prior và vẽ những đường thẳng suy ra bằng code như trước. Kết quả được thể hiện ở bên phải [**HÌNH 8.3**](#f3). Vài slope vẫn còn mạnh không phù hợp. Nhưng quan trọng, đây là tập prior tốt hơn. Nhìn vào posterior:

<b>code 8.6</b>
```python
post = m8_1.sample_posterior(random.PRNGKey(1), p8_1, (1000,))
print_summary({k: v for k, v in post.items() if k != "mu"}, 0.89, False)
```
<samp>       mean   std  median   5.5%  94.5%    n_eff  r_hat
    a  1.00  0.01    1.00   0.98   1.02   931.50   1.00
    b  0.00  0.06    0.00  -0.08   0.10  1111.63   1.00
sigma  0.14  0.01    0.14   0.13   0.15   949.29   1.00</samp>

Thực ra nhìn chung không có quan hệ giữa độ gồ ghề địa hình và log GDP. Tiếp theo chúng ta sẽ xem cách chia tách các lục địa.

<div class="alert alert-info">
<p><strong>Luyện tập cho lúc cần nó.</strong> Bài tập trong <a href="#f3"><strong>HÌNH 8.3</strong></a> thực ra không cần thiết trong ví dụ này, bởi vì có đủ data, và mô hình là đủ đơn giản, mà ngay cả prior tệ cũng bị mất tác dụng. Bạn thậm chí có thể sử dụng prior phẳng (đừng!), và nó vẫn tốt. Nhưng chúng ta luyện tập tốt thực hành này không phải vì nó luôn quan trọng. Mà là, chúng ta luyện tập làm việc này tốt để chúng ta chuẩn bị sẵn khi cần nó. Không ai nói rằng đeo đai bụng là sai lầm, chỉ bởi vì bạn không bị tai nạn.</p></div>

### 8.1.2 Thêm biến chỉ điểm là không đủ

Điều đầu tiên cần nhận ra là chỉ thêm biến chỉ điểm (indicator) cho quốc gia Châu Phi, `cont_africa` ở đây, sẽ không hiện ra hiện tượng đảo chiều của slope. Mặc dù bạn nên thử fit mô hình để tự chứng minh điều đó. Tôi sẽ đi qua phần này bằng bài tập so sánh mô hình đơn giản, để bạn bắt đầu có những ví dụ thực dụng của các khái niệm bạn đã tích luỹ từ những chương trước. Chú ý rằng so sánh mô hình ở đây không phải để chọn mô hình. Những kiến thức khoa học đã chọn ra mô hình liên quan. Thay vào đó nó là đo lường ảnh hưởng của sự khác nhau của mô hình khi quan tâm đến nguy cơ overfitting.

Để xây dụng mô hình cho phép quốc gia trong và ngoài Châu Phi để có những intercept khác nhau, chúng ta cần tuỳ chỉnh mô hình cho $\mu_i$ để trung bình được đặt điều kiện trên lục địa. Cách thuận tiện để làm điều này là chỉ cần thêm một số hạng khác vào mô hình tuyến tính:

$$\mu_i = \alpha +\beta(r_i -\bar{r}) + \gamma A_i$$

Trong đó $A-i$ là `cont_affrica`, một biến chỉ điểm 0/1. Nhưng đừng theo quy tắc này. Thực vậy, quy tắc này thông thường là một ý tưởng tệ. Tôi đã bỏ ra nhiều năm để nhận ra điều này, và tôi cố gắng cứu bạn khỏi nỗi kinh hoàng tôi đã gặp. Vấn đề là, tổng quát, chúng ta cần prior cho $\gamma$. Được rồi, chúng ta có thể làm prior. Nhưng những gì prior làm là nói cho mô hình biết $\mu_i$ cho một quốc gia trong Châu Phi là bất định hơn, trước khi thấy data, hơn $\mu_i$ ngoài Châu Phi. Và nó nghe hợp lý. Đây cũng giống như vấn đề chúng ta gặp ở Chương 4, khi tôi giới thiệu biến phân nhóm.

Đây là một giải pháp đơn giản: Quốc gia ở Châu Phi sẽ có một intercept và những quốc gia ngoài Châu Phi cũng như vậy. Lúc này $\mu_i$ sẽ có dạng:

$$\mu_i = \alpha_{\Tiny CID[i]} + \beta(r_i - \bar{i}) $$

trong đó $CID$ là biến chỉ số, ID của lục địa. Nó nhận giá trị 0 cho quốc gia ở Châu Phi và 1 cho quốc gia khác. Điều này có nghĩa có 2 tham số, $\apha_1$ à $\alpha_2$, mỗi một cho từng giá trị chỉ số độc nhất. Ký hiệu $CID[i]$ nghĩa là giá trị $CID$ ở hàng $i$. Tôi dùng ký hiệu ngoặc vuông cho biến chỉ số, bởi vì nó dễ hơn để đọc hơn thêm một dòng nằm dưới, $\alpha_{CID_i}$. Chúng ta có thể xây dựng biến chỉ số này như sau:

<b>code 8.7</b>
```python
# make variable to index Africa (0) or not (1)
dd["cid"] = jnp.where(dd.cont_africa.values == 1, 0, 1)
```



## <center></center>8.2 Tính đối xứng của tương tác<a name="a2"></a>
## <center></center>8.3 Tương tác liên tục<a name="a3"></a>
## <center></center>8.4 Tổng kết<a name="a4"></a>


<a name="f1"></a>![](/assets/images/fig -.svg)
<details class="fig"><summary></summary>
{% highlight python %}{% endhighlight %}</details>

---

<details><summary>Endnotes</summary>
<ol start="" class='endnotes'>
    <li></li>
</ol>
</details>

<details class="practice"><summary>Bài tập</summary>
<p>Problems are labeled Easy (E), Medium (M), and Hard (H).</p>
</details>