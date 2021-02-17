# Giới thiệu Bayesian statistics

Trong vòng những năm gần đây, ta thấy ngày càng có nhiều ứng dụng dựa trên kỹ thuật *Machine Learning*, đặc biệt là trong lĩnh vực search engine, thương mại điện tử, quảng cáo, mạng xã hội, v,v,.. Những ứng dụng này tập trung vào độ chính xác trong dự đoán và cần lượng data lớn ( tính bằng tetrabyte). Thực tế đó là nền tảng của những ông Tech lớn như Google, Facebook,..

Tuy nhiên, đa số những ứng dụng này là *blackbox*, nghĩa là không thể diễn giải được. Ví dụ như những mô hình quảng cáo hướng đối tượng, người quản lý không biết mô hình hoạt động như thế nào, chỉ cần biết là nó cho kết quả tốt là được.  
Một nhược điểm thứ  là những ứng dụng này cần rất nhiều data. Ví dụ như trong ứng dụng quảng cáo hướng đối tượng, họ cần hàng triệu người sử dụng để tạo ra sản phẩm mô hình quảng cáo.  

Những giới hạn này làm cho việc tạo mô hình (modeling) khó hơn ở những lĩnh vực ít data hoặc chuyên sâu. Nó cũng có thể gây tác dụng phụ trong bối cảnh liên quan tới sinh mạng và luật pháp như y học hoặc bảo hiểm. Ở đây, mô hình dự đoán với độ chính xác phải kèm theo độ tin cậy để ước lượng nguy cơ.  
Ví dụ: Chúng ta phải ước lượng được tính bất định khi đưa ra chẩn đoán bệnh dương tính cho con người.

**Bayesian** là một phương pháp phân tích có thể vượt qua những nhược điểm này. Kỹ thuật bắt đầu với thiết lập thông tin ban đầu (prior) vào hệ thống cần mô hình hoá, sau đó kết hợp với data thu thập được và giả định về phân phối của data (likelihood) để tạo ra mô hình bị ràng buộc bởi thông tin ban đầu và data, dưới giả định phân phối likelihood.  
Từ mô hình đó, ta có thể dùng để dự đoán và kèm theo đó là một độ tin cậy được biểu diễn bằng phân phối.  
Phương pháp Bayesian hoạt động tốt khi có dữ liệu ít, kèm theo khái niệm độ tin cậy, và có thể diễn giải được.

**Probabilistic programming** là một dạng lập trình bậc cao, có ưu điểm ẩn đi các phép tính toán phức tạp trong phương pháp bayesian.

# Học phân tích Bayes
Để học phân tích Bayes, tôi có đọc và dịch sách Statistical Rethinking 2nd của tác giả Riichard McElreath. Đây là một cuốn sách rất được nhiều người đề nghị, bởi nội dung được trình bày rất rõ ràng và chi tiết, độ khó được nâng dần theo bậc thang, phù hợp với tất cả mọi người.  
Các bạn hãy ủng hộ tác giả thông qua sản phẩm trí tuệ của ông: [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

Các bài dịch của tôi, các bạn vui lòng vào mục lục có ở menu trên cùng. (Project đang dần hoàn thiện)

Tuy nhiên, do sách dùng Rcode và tôi cũng không chuyên nó, nên trong bài dịch tôi học Bayes bằng Python.  
Lợi điểm của Python là code dễ học, dễ đọc, dễ hiểu, linh hoạt cao, có thể được tích hợp vào các sản phẩm thương mại hoá.

Trong python có rất nhiều framework để viết probabilistic programming: như `tensorflow`, `pytorch`, `pyro`, `numpyro`, `pystan`, `probflow`, `pymc3`,... Tôi chọn `numpyro` vì dev viết `numpyro` trong đó có người Việt. Thực ra code `numpyro` nhìn chung tối ưu hơn những framework khác. Tuy nhiên bạn chỉ dùng được nó với hệ điều hành Macos hoặc Linux thôi. Nếu bạn dùng Wins thì có thể chuyển qua những framework khác, trong đó `pymc3` khá nổi tiếng.

Code and examples:
- R package: [rethinking](https://github.com/rmcelreath/rethinking) (github repository)
- R code examples from the book: [code.txt](http://xcelab.net/rmpubs/sr2/code.txt)
- Data copy lại từ sách rethinking: [download data](https://github.com/rmcelreath/rethinking/tree/master/data)
- Book examples in [Stan+tidyverse](https://vincentarelbundock.github.io/rethinking2/)
- brms + tidyverse conversion [here](https://bookdown.org/content/4857/)
- PyMC3 code examples: [PyMC repository](https://github.com/pymc-devs/resources/tree/master/Rethinking_2)
- [Pyro](https://fehiepsi.github.io/rethinking-pyro)
- [NumPyro](https://fehiepsi.github.io/rethinking-numpyro/)
- TensorFlow Probability [notebooks](https://github.com/ksachdeva/rethinking-tensorflow-probability)
- [Julia & Turing](https://github.com/StatisticalRethinkingJulia) examples (both 1st and 2nd edition)
- [R-INLA](https://akawiecki.github.io/statistical_rethinking_inla/) examples

# Bắt đầu với Probabilistic programming

## Cài đặt:
- [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html): Phiên bản nhẹ của anaconda, một ứng dụng để download python và các packages, tạo môi trường biệt lập để làm việc cho từng project.
- Sau khi cài đặt miniconda xong, bạn sẽ làm việc trong môi trường command promt trong windows hoặc terminal trong macos.
    - command promt: search cortana -> gõ `cmd` -> run
    - terminal: search spotlight -> gõ `terminal` -> run
    - gõ trong cmd `conda --version` để kiểm tra cài đặt thành công.
- Cài đặt channel conda-forge: [conda-forge](https://conda-forge.org) là một channel của conda chứa hầu như các packages có trong python, mà khi bạn cài đặt thì không cần quan tâm đến sự tương thích giữa các packages với nhau.
```
conda config --add channels conda-forge 
conda config --set channel_priority strict 
```
- Tạo môi trường mới với \<env_name\> là tên bạn chọn với python=3.8: 
```
conda create --name <env_name> python=3.8
```
Ví dụ: `conda create --name myenv python=3.8`
- Kích hoạt env:
```
conda activate <env_name>
```
- Cài đặt các packages sẽ dùng trong site này (sẽ update dần)
```
conda install pytorch numpyro arviz jupyterlab pandas causalgraphicalmodels daft
```
- Mở Jupyter và code python trong đó
```
jupyter lab
```

## Học Python cơ bản
Bạn cần phải biết Python. Nếu bạn mới bắt đầu thì có thể tìm hiểu cơ bản qua [tutorials trong python doc.](https://docs.python.org/3.7/tutorial/index.html)

# Tác giả:
- [Richard McElreath](https://twitter.com/rlmcelreath)
- [Du Phan](https://twitter.com/fehiepsi)

# Chúc các bạn thành công