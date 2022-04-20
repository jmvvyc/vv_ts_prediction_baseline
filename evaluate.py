#!/usr/bin/env python
# -*- coding:utf-8 -*-
import datetime
import getopt
import logging
import random
import math
import multiprocessing as mp
import os
import sys
import traceback
import imageio
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tqdm
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

import poolcontext

mpl.use('Agg')  # put this BEFORE import matplotlib.pyplot

# https://developpaper.com/installing-chinese-font-on-linux/
plt.rcParams['font.sans-serif'] = ['SimHei']
# https://stackoverflow.com/questions/58361594/matplotlib-glyph-8722-missing-from-current-font-despite-being-in-font-manager#answer-64376187
plt.rc('axes', unicode_minus=False)
# Note that matplotlib.pyplot.tight_layout() will only adjust the subplot params when it is called. In order to perform this adjustment each time the figure is redrawn, you can call fig.set_tight_layout(True), or, equivalently, set the figure.autolayout rcParam to True.
plt.rcParams["figure.autolayout"] = True

# !pip install scikit-learn==0.24

my_dpi = 144  # https://www.infobyip.com/detectmonitordpi.php
fig_dpi = my_dpi * 2
# since python 3.7, https://stackoverflow.com/questions/4374455/how-to-set-sys-stdout-encoding-in-python-3#answer-52372390
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)
mp_logger = mp.get_logger()
mp_formatter = logging.Formatter(
    '%(process)d,%(processName)s\t%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(mp_formatter)
mp_logger.addHandler(handler)
mp_logger.setLevel(logging.INFO)
png_out_path = "png"
tsv_out_path = "out"
date_format = "%Y-%m-%d"
max_h_val = 7
out_path_comm_prefix = "comp_2022"
sep_regex = r"[\t,]"
test_sample_cnt = 7
gif_delay = 0.4
y_col = "cid_day_vv_t"
submit_out_cols = ["cid_t", "nth_day", "yhat"]
data_path = "in/rank_a_data.tsv"


installed_fonts = set([f.name for f in fm.fontManager.ttflist])
logger.info("%d installed font is '%s'", len(
    installed_fonts), list(installed_fonts))


def get_mMAPE(actual_df, submit_df):
    global logger
    mMAPE = sys.float_info.max
    submit_df = submit_df.rename(columns={submit_df.columns[2]: 'yhat'})
    try:
        calc_df = submit_df[["cid_t", "nth_day", "yhat"]].merge(
            actual_df[["cid_t", "nth_day", "y"]], on=["cid_t", "nth_day"], how="inner")
        valid_cnt = calc_df.shape[0]
        actual_cnt = actual_df.shape[0]
        if valid_cnt < actual_cnt:
            raise ValueError(
                "submit has only %d EFFECTIVE rows while %d rows is required" % (valid_cnt, actual_cnt))
        cid_grp_df = calc_df.groupby(["cid_t"])
        mape_li = []
        for _, cur_grp_df in cid_grp_df:
            cur_actual = cur_grp_df["y"].astype(float)
            cur_submit = cur_grp_df["yhat"].astype(float)
            cur_mape = mean_absolute_percentage_error(cur_actual, cur_submit)
            mape_li.append(cur_mape)
        mMAPE = np.mean(mape_li)
    except Exception as e:
        logger.error("'%s'\n%d submit_df\n%s\n%d actual_df\n%s", str(e),
                     submit_df.shape[0], submit_df.to_string(index=False), actual_df.shape[0], actual_df.to_string(index=False))
    return mMAPE


def usage():
    pass


def get_forecast(cur_cid_t, cur_data_df, x_col, y_col, fit_cols):
    global mp_logger
    global max_h_val
    global date_format
    global tsv_out_path
    global my_dpi
    global fig_dpi
    global submit_out_cols
    global test_sample_cnt
    cur_desc = ""
    try:
        cur_fit_data = cur_data_df[cur_data_df["cid_t"]
                                   == cur_cid_t]
        if cur_fit_data.empty:
            mp_logger.warning("%d cur_data_df\n%s\nEMPTY fit data",
                              cur_data_df.shape[0], cur_data_df.head().to_string(index=False))
            return []
        cur_fit_data = cur_fit_data.rename(
            columns={x_col: "ds", y_col: "y"})[fit_cols]
        cur_start_ds = cur_fit_data["ds"].min()
        cur_cid_t = cur_fit_data["cid_t"].unique().tolist()[0]
        cur_samp_cnt = cur_fit_data.shape[0]
        cur_desc = "cid_t '%d' %d samples" % (
            cur_cid_t,  cur_samp_cnt)
        mp_logger.info("\t%s %d cur_fit_data from '%s' to '%s'\n%s", cur_desc, cur_samp_cnt, cur_fit_data["ds"].min(), cur_fit_data["ds"].max(),
                       cur_fit_data.sort_values(by=["ds"], ascending=[True]).to_string(index=True))
        fit_nan_sel = cur_fit_data.isna().any(axis=1)
        fit_nan_cnt = cur_fit_data[fit_nan_sel].shape[0]
        if fit_nan_cnt > 0:
            mp_logger.info("%s %d nan in cur_fit_data\n%s",
                           cur_desc, fit_nan_cnt, cur_fit_data[fit_nan_sel].to_string(index=False))
        m = Prophet(changepoint_prior_scale=0.01)
        # https://stackoverflow.com/questions/50582168/pandas-get-all-columns-that-have-constant-value#answer-50582195
        constant_cols = cur_fit_data.columns[cur_fit_data.eq(
            cur_fit_data.iloc[0]).all()].tolist()
        mp_logger.debug("%s have %d constant_cols '%s'",
                        cur_desc, len(constant_cols), "|".join(constant_cols))
        if constant_cols:
            mp_logger.info("%s fit data have %d constant_cols '%s'", cur_desc, len(
                constant_cols), "|".join(constant_cols))
        train_sample_cnt = cur_fit_data.shape[0]
        m.fit(cur_fit_data)
        # mp_logger.debug("%s %d test_data\n%s" ,
        # cur_desc, test_data.shape[0], test_data.to_string(index=False))
        # By default it will also include the dates from the history
        future = m.make_future_dataframe(
            periods=7, freq="D", include_history=False)
        forecast = m.predict(future)
        mp_logger.debug("%s %d forecast '%s'\n%s",
                        cur_desc, forecast.shape[0], "|".join(list(forecast.columns)), forecast.to_string(index=False))
        # save forecast to file
        cur_fcst_data_path = "%s/%s_%s_%dtrain_%d_forecast.tsv" % (
            tsv_out_path, out_path_comm_prefix, cur_cid_t, train_sample_cnt, test_sample_cnt)
        forecast["cid_t"] = cur_cid_t
        forecast["dummy_start_ds"] = cur_start_ds
        forecast["nth_day"] = forecast["ds"].dt.date - \
            forecast["dummy_start_ds"].dt.date
        forecast["nth_day"] = forecast["nth_day"].apply(
            lambda td_obj: (td_obj + datetime.timedelta(days=1)).days)
        mp_logger.debug("%s %d forecast rows ds columns\n%s", cur_desc, forecast.shape[0], forecast[[
                        "nth_day", "dummy_start_ds", "ds"]].to_string(index=False))
        forecast[submit_out_cols].to_csv(
            cur_fcst_data_path, sep="\t", index=False)
        mp_logger.info("%s forecast data path '%s'", cur_desc,
                       os.path.abspath(cur_fcst_data_path))
        # fig = m.plot(forecast)
        fcst_plt, fcst_ax = plt.subplots()
        xticklabels = [cur_dt.strftime(date_format)
                       for cur_dt in cur_fit_data['ds'].tolist()]
        # mp_logger.debug("%s %d forecast\n%s",
        # cur_desc, forecast.shape[0], forecast.to_string(index=False))
        fcst_ax.plot(forecast['ds'], forecast['yhat'])
        y_lbl = "预测值"
        fcst_ax.lines[-1].set_label(y_lbl)
        fcst_ax.fill_between(forecast['ds'], forecast['yhat_lower'],
                             forecast['yhat_upper'], color='b', alpha=.3)
        fcst_ax.set_xticklabels(xticklabels, rotation=45,
                                ha="right", rotation_mode="anchor")
        fcst_ax.legend()
        fcst_ax.set_title("%s" % (cur_desc))
        plt.tight_layout()  # make rotated labels fitting into canvas
        cur_fcst_plt_path = "%s/cid_t_%d_forecast.png" % (
            png_out_path, cur_cid_t)
        fcst_plt.savefig(cur_fcst_plt_path, figsize=(
            1920/my_dpi, 1200/my_dpi), dpi=fig_dpi, bbox_inches='tight')
        plt.close()
        mp_logger.info("%s", os.path.abspath(cur_fcst_plt_path))
        comp_fig = m.plot_components(forecast)
        cur_comp_fig_path = "%s/cid_t_%d_comp_forcast.png" % (
            png_out_path, cur_cid_t)
        mp_logger.info("%s", os.path.abspath(cur_comp_fig_path))
        comp_fig.savefig(cur_comp_fig_path)
        return [cur_cid_t,
                train_sample_cnt, test_sample_cnt,
                cur_fcst_plt_path, cur_fcst_data_path]
    except Exception as e:
        mp_logger.error("%s\n%s\n\n%s",
                        cur_desc, str(e), traceback.format_exc())
        return []


def main():
    global logger
    global out_path_comm_prefix
    global png_out_path
    global tsv_out_path
    global sep_regex
    global gif_delay
    global submit_out_cols
    global data_path
    global y_col
    if not os.path.exists(png_out_path):
        os.makedirs(png_out_path)
    if not os.path.exists(tsv_out_path):
        os.makedirs(tsv_out_path)
    logger.info("python path '%s', version '%s'", sys.executable, sys.version)
    thread_n = 1
    data_path_set = False
    log_level = "INFO"
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "h", ["help", "thread=", "input=", "log-level="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--thread"):
            thread_n = int(a)
            logger.info("thread_n=%d", thread_n)
        elif o in ("-i", "--input"):
            data_path = a
            logger.info("data_path='%s'", data_path)
            data_path_set = True
        elif o in ("--log-level"):
            log_level = a.upper()
            logger.setLevel(log_level)
            mp_logger.setLevel(log_level)
            logger.info("logger level '%s'", log_level)
        else:
            assert False, "unhandled option"
    # ...
    if len(args) > 0:
        if not data_path_set:
            data_path = args[0]
            logger.info("data path '%s'", os.path.abspath(data_path))
        else:
            logger.error(
                "-i|--input and positional argument #1 are MUTUALLY exclusive...")
            sys.exit(1)

    # ...
    # logger.debug("mpl cachepath '%s'", mpl.get_cachedir())
    logger.debug("python path '%s', version '%s'",
                 sys.executable, sys.version)
    logger.debug("pandas '%s', version %s", pd.__file__, pd.__version__)
    logger.debug("numpy '%s', version %s", np.__file__, np.__version__)
    logger.debug("sklearn '%s', version %s",
                 sklearn.__file__, sklearn.__version__)
    logger.debug("matplotlib '%s', version %s",
                 mpl.__file__, mpl.__version__)

    start_dt = datetime.datetime.now()
    abs_data_path = os.path.abspath(data_path)
    raw_data_df = pd.read_csv(data_path, sep="\t", encoding="utf-8")
    data_df = raw_data_df[raw_data_df[y_col] != 0] # cid_day_vv_t列为0的为待预测值，排除
    rand_day_delta = random.randint(1, 3*365)
    rand_start_dt = start_dt - datetime.timedelta(days=rand_day_delta)
    logger.debug("rand_start_dt '%s', rand_day_delta %d",
                 rand_start_dt.strftime(date_format), rand_day_delta)
    # 随机选1个日期作为起始日期， Prophet模型要求x轴为datetime或者date
    data_df["ds"] = data_df["nth_day"].apply(
        lambda cur_nth_d_val:  rand_start_dt + datetime.timedelta(days=cur_nth_d_val))
    end_dt = datetime.datetime.now()
    data_df = data_df.astype({"cid_t": int, "nth_day": int})
    logger.info("%d '%s' data_df '%s' readed from '%s' in '%s'\n%s",
                data_df.shape[0], list(
                    data_df.columns), data_df.dtypes.to_string(),
                abs_data_path, end_dt-start_dt, data_df.head().to_string(index=False))
    fit_cols = ["nth_day", "ds", "y", "cid_t", "channelId_t", "leader_t",
                "kind_t", "seriesId_t", "seriesNo"]
    y_lbl = "脱敏值"
    logger.info("fitting y column is '%s'(%s)", y_col, y_lbl)
    x_col = "ds"
    logger.info("fitting x column is '%s'", x_col)
    id_col = "cid_t"
    ins = []
    args_list = []
    cid_t_li = data_df["cid_t"].unique().tolist()
    for cur_cid_t in cid_t_li:
        args_list.append(
            (cur_cid_t, data_df, x_col, y_col, fit_cols))
    chunksize = int(math.ceil(len(args_list)/float(thread_n)))
    pool_res = []
    job_star_dt = datetime.datetime.now()
    with poolcontext.poolcontext(thread_n) as pool:
        pool_res = pool.starmap(get_forecast, tqdm.tqdm(
            args_list, total=len(args_list)), chunksize=chunksize)
    job_end_dt = datetime.datetime.now()
    logger.info("%d jobs with %d batch in %s", len(args_list),
                chunksize, str(job_end_dt - job_star_dt))
    for cur_ret_row in pool_res:
        if cur_ret_row:
            ins.append(cur_ret_row)
    eval_res_cols = ["cid_t", "train_sample_cnt", "test_sample_cnt",
                     "forecast_plot_path", "forecast_data_path"]
    eval_aggr_df = pd.DataFrame(ins, columns=eval_res_cols)
    topic_desc = "cid_ts_vv_eval"
    # make submession sample
    submit_df = pd.DataFrame()
    chosen_cids = eval_aggr_df["cid_t"].unique().tolist()
    for _, cur_cid_t in tqdm.tqdm(enumerate(chosen_cids)):
        cur_fcst_data_path = eval_aggr_df.loc[eval_aggr_df["cid_t"]
                                              == cur_cid_t]["forecast_data_path"].values[0]
        cur_fcst_df = pd.read_csv(
            cur_fcst_data_path, sep="\t", encoding="utf-8")
        cur_fcst_df[id_col] = int(cur_cid_t)
        # logger.debug("#%d %d cur_fcst_df %d columns '%s'\n%s",
        #             i+1, cur_fcst_df.shape[0], cur_fcst_df.shape[1],
        #             "|".join(list(cur_fcst_df.columns)),
        #             cur_fcst_df.head().to_string(index=False))
        submit_df = pd.concat([submit_df, cur_fcst_df], axis=0)
    submit_out_path = "%s/%s_%s_all_submission.tsv" % (
        tsv_out_path,  out_path_comm_prefix, topic_desc)
    submit_df[submit_out_cols].to_csv(
        submit_out_path, sep="\t", encoding="utf-8", index=False)
    logger.info("ALL submissions path '%s'",
                os.path.abspath(submit_out_path))
    # save gif
    if not eval_aggr_df.empty:
        all_images = []
        for _, cur_cv_row in eval_aggr_df.iterrows():
            cur_png_path = cur_cv_row["forecast_plot_path"]
            all_images.append(imageio.imread(cur_png_path))
        if all_images:
            gif_path = "%s/%s_%s_movie.gif" % (
                png_out_path, out_path_comm_prefix, topic_desc)
            imageio.mimsave(gif_path, all_images, duration=gif_delay)
            logger.info("ALL chosen data gif path '%s'",
                        os.path.abspath(gif_path))


if __name__ == "__main__":
    main()
