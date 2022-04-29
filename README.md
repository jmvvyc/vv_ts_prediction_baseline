# 第三届“马栏山杯”国际音视频算法大赛-节目播放量预测"赛题简介
* [赛题链接](http://challenge.ai.mgtv.com/contest/detail/13) 
* **说明**： 节目的播放量预测能帮助平台多项业务开展。比如提前进行服务资源准备以应对流量变化情况、辅助在线广告库存预估等。本赛题通过利用真实的节目播放数据，脱敏处理后，尽可能还原节目播放量预测的问题原型，使得选手们可以探索真实业务场景的有效解决方案，问题具体如下：基于前![](https://latex.codecogs.com/svg.image?T)天的节目播放数据、节目属性和时间上下文，预测未来一段时间节目的播放量  
**Introduction**: Prediction of program on-demand ![](https://latex.codecogs.com/svg.image?VideoVisit) count plays an important role in the field of Internet data mining.   
Programs with high on-demand (especially movies and TV series) can improve the coverage of online advertisements.   
Prediction based on time series has been applied concretely in advertising business expansion, *ContentDistributionNetwork* bandwidth optimization etc.   
For example, the overall ![](https://latex.codecogs.com/svg.image?VideoVisit) of the target drama program is projected based on the actual ![](https://latex.codecogs.com/svg.image?Video&space;Visit(VV)) of ![](https://latex.codecogs.com/svg.image?T) days previously, so that the user audience's preference for the target drama series can be known in advance through the previously collected data, hence a more accurate end-user rating for the target program can be known from an early stage. 
* **目的**：致力将平台端预测未上线节目分时播放量![](https://latex.codecogs.com/svg.image?Video&space;Visit(VV))以提前准备*CDN*等资源的业务线上流程用该赛题完整地模拟出来，使得*TOP*选手的方案可成功用于线上实际业务。  
* **Objective**: Aimed at fully simulating the online program’s ![](https://latex.codecogs.com/svg.image?VideoVisit) (VV)’ ebb and flow life cycle of the [*MGTV.com*](www.mgtv.com) platform,  predicting the overall and dayly ![](https://latex.codecogs.com/svg.image?VideoVisit) count of video programs that are yet to release to prepare *CDN* and other resources in advance, so that the TOP players' solutions can be successfully applied to boost users’ watching experiences .

# 数据说明
 * 评价指标： ![](https://latex.codecogs.com/svg.image?N)个节目/合集的测试集![](https://latex.codecogs.com/svg.image?mMAPE)指标平均值， 即
![](https://latex.codecogs.com/svg.image?mMAPE&space;=&space;\frac{&space;\sum_{i=1}^{N}&space;\frac{&space;\sum_{t=1}^{J}&space;\left|&space;\frac{\hat{y}&space;-&space;y_{t}}{y_{t}}&space;\right|&space;}{J}&space;}{N})
其中 ![](https://latex.codecogs.com/svg.image?\hat{y})是第![](https://latex.codecogs.com/svg.image?t)天的![](https://latex.codecogs.com/svg.image?VV)预测值, ![](https://latex.codecogs.com/svg.image?y_t) 是第![](https://latex.codecogs.com/svg.image?t)天的真实![](https://latex.codecogs.com/svg.image?VV)值， ![](https://latex.codecogs.com/svg.image?N)为预测节目总数， ![](https://latex.codecogs.com/svg.image?J)为预测的最大天数/*时间窗口*

* 赛题数据列说明  
以下数据列均经过脱敏操作  

|序号|列名|说明|示例|备注|
|:---:|:---:|:---:|:---:|:---:|
|1|*nth_day*|日期序数|![](https://latex.codecogs.com/svg.image?1)|脱敏数据,![](https://latex.codecogs.com/svg.image?1)表示第![](https://latex.codecogs.com/svg.image?1)天|
|2|*cid_day_vv_t*|当天节目播放数|![](https://latex.codecogs.com/svg.image?355628)|已脱敏，该列为![](https://latex.codecogs.com/svg.image?0)时表示**待**预测值|
|3|*is_holiday*|当天是否节假日|![](https://latex.codecogs.com/svg.image?False)|真实值|
|4|*date_has_update*|当天是否有更新|![](https://latex.codecogs.com/svg.image?True)|真实值|
|5|*vv_vid_cnt*|截止当天累计播放视频个数|![](https://latex.codecogs.com/svg.image?4)|真实值, 包括当天统计值|
|6|*online_vid_cnt*|截止当天累计上线视频个数|![](https://latex.codecogs.com/svg.image?5)|真实值, 包括当天统计值|
|7|*week_day*|当天星期几|![](https://latex.codecogs.com/svg.image?3)|真实值|
|8|*seriesNo*|第几季节目|![](https://latex.codecogs.com/svg.image?9)|脱敏数据|
|9|*cid_t*|节目标识ID|![](https://latex.codecogs.com/svg.image?29)|脱敏数据|
|10|*seriesId_t*|节目所属IP标识ID|![](https://latex.codecogs.com/svg.image?27)|脱敏数据|
|11|*channelId_t*|节目所属频道标识ID|![](https://latex.codecogs.com/svg.image?1)|脱敏数据|
|12|*leader_t*|节目主演标识ID|![](https://latex.codecogs.com/svg.image?34,5,41,13,25,57,29,31)|脱敏数据 多个用英文","分隔|
|13|*kind_t*|节目类型标识ID|![](https://latex.codecogs.com/svg.image?9,3,45)|脱敏数据 多个用英文","分隔|

* A榜：![](https://latex.codecogs.com/svg.image?9368)个节目最近 *T*天的分天VV/视频播放数，baseline ![](https://latex.codecogs.com/svg.image?mMAPE=41\%),
* [A榜赛题数据链接](http://challenge.ai.mgtv.com/contest/detail/13?locale=zh) 
-  &ensp; &ensp; 老数据文件rank_a_data.tsv *MD5*:**2244b45832f620cb78b344a95c5f84e5**
-  &ensp; &ensp; 补充数据文件 rank_a_supp_data.tsv *MD5*:**09a3f99fa1a97bb637df1e0c51b402f3**
* B榜：待公布

### 注意事项
* *A*榜数据在*A*榜开放之后对参赛选手开放下载，*B*榜测试集数据*B*榜开放之后对参赛选手开放下载；
* 竞赛过程中不能使用其他第三方数据
* 除公开的数据集外，本大赛涉及的数据集（含合集标签数据）版权归芒果TV所有，选手不能将其泄露或用于本大赛外其他用途

### 提交样例
提交文件为tsv格式(不同的字段使用制表符”_**\t**_”分隔)，**必须**包含固定文件头， 格式如下：

|![](https://latex.codecogs.com/svg.image?cid\\_t)|![](https://latex.codecogs.com/svg.image?nth\\_day)|![](https://latex.codecogs.com/svg.image?VV)|
|:---:|:---:|:---:|
|1|1|0.1|
|1|2|3.5|
|2|1|0.2|
|2|2|4.6|
|2|3|8.6|

 [**提交数据样例**](out/reversion_rank_a_submission.tsv)  
 
## FAQ
* B榜数据是预测A榜相同节目往后推![](https://latex.codecogs.com/svg.image?7)天的![](https://latex.codecogs.com/svg.image?VV)吗
  * &emsp; 是的。 B榜赛题会包括A榜赛题的答案， 并要求预测**同样**的节目往后再推![](https://latex.codecogs.com/svg.image?7)天的每日![](https://latex.codecogs.com/svg.image?VV)
* 为什么对于相同的*cid_t*值， *leader_t*, *kind_t*都是常量？
  * &emsp; 这两列为脱敏后的节目标签， 属于ID类特征， 虽然对于同一个节目是常量， 但是**不同**节目之间不同， 待挖掘。
  
  
### 算力要求
 1. 内存使用不超过64G 未满足以上算力限制的参赛队伍，大赛官方有权将最终总成绩判定无效，排名由后一名依次递补。
 
### 文件夹说明
  * in文件夹: 赛题数据
  * png文件夹：输出汇总预测结果GIF以及分节目预测走势图/[CV](https://facebook.github.io/prophet/docs/diagnostics.html)图/component plot，使用[cli小节](#cli)命令运行完毕以后自动生成
  * out文件夹：输出预测指标汇总统计tsv文件
  
## 运行基准测试
### 依赖
 * python 3.7+
 * pip install -i https://mirrors.aliyun.com/pypi/simple/ -r evaluate_requirements.txt

### cli

|文件名|说明|
|:---:|:---:|
|evaluate.py|基准测试代码|
|poolcontext.py|多个模型多进程并行训练依赖库|
|evaluate_requirements.txt|全部依赖python库|
|out/comp_2022_cid_ts_vv_eval_all_submission.tsv|运行evaluate.py以后生成的Baseline提交样例文件|

ZSH下输入
 `time nohup python3 ./evaluate.py --thread=4 &> evaluate.log &&
 tail -F evaluate.log`  

### 代码测试环境
 - OS: Ubuntu 18.04.03 LTS 
 - 内存：128G内存 实际内存消耗远低于64G,  依赖于具体的cli --thread参数
 - CPU: 32核64位Intel CPU
 
