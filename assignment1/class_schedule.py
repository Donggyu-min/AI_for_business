# -*- coding: utf-8 -*-
import pandas as pd


def period_converter(start, end):
    # 수업 시작/끝시간을 받아 중간을 period로 채움, 네트워크 부하 기준으므로 보수적 측정

    st_period = 0
    end_period = 0

    if start < 10.5:    st_period = 1
    elif start < 13:    st_period = 2
    elif start < 14.5:  st_period = 3
    elif start < 16:    st_period = 4
    elif start < 17.5:  st_period = 5
    elif start < 19:    st_period = 6
    else:               st_period = 7

    if end <= 10.5:    end_period = 1
    elif end <= 13:    end_period = 2
    elif end <= 14.5:  end_period = 3
    elif end <= 16:    end_period = 4
    elif end <= 17.5:  end_period = 5
    elif end <= 19:    end_period = 6
    else:              end_period = 7

    end_period = end_period+1
    period_range = list(range(st_period, end_period))
    return period_range


def time_converter(time_string):
    # CRShour을 week, start_time, end_time으로 구분한 후 list로 붙임(week,period)
    time_string = time_string.split('\r\n')
    wp_list = []
    for i in range(len(time_string)):
        week = time_string[i].split(' ')[0]

        start_time = time_string[i].split(' ')[1].split('~')[0].split(':')
        start_hour = float(start_time[0]) + (float(start_time[1])/60)

        end_time = time_string[i].split(' ')[1].split('~')[1].split(':')
        end_hour = float(end_time[0]) + (float(end_time[1])/60)

        period_range = period_converter(start_hour, end_hour)
        for j in range(len(period_range)):
            wp = (week, period_range[j])
            wp_list.append(wp)
    wp_list = list(set(wp_list))
    return wp_list


def credit_converter(L_E_C):
    # (수업시간:실험시간:학점)형태의 L_E_C를 credit에 따라 over1 or cred3으로 구분
    credit = float(L_E_C.split(':')[2])
    if credit == 3.0:
        credit = 'cred3'
    elif credit >= 1.0:
        credit = 'over1'
    else:
        credit = 'no_credit'
    return credit


def counter(df):
    # wp_list를 multiple row로 만든 뒤, wp_list & L_E_C를 기준으로 groupby하여 처리
    df = df.explode('wp_list').reset_index(drop=True)
    grouped_df = df[['wp_list', 'L_E_C', 'Rstu']].groupby(by=['wp_list', 'L_E_C'], as_index=False)
    sum_df = pd.merge(grouped_df.count(), grouped_df.sum()['Rstu'], left_index=True, right_index=True)
    sum_df = sum_df.rename({'Rstu_x':'num_class', 'Rstu_y':'num_Rstu'}, axis='columns')
    # print(sum_df)
    return sum_df


if __name__ == '__main__':
    df = pd.read_excel("Fall200901.xlsx")
    df = df.dropna(subset=['CRShour']).reset_index(drop=True)
    df['wp_list'] = 0
    for idx, row in df.iterrows():
        row['wp_list'] = time_converter(row['CRShour'])
        row['L_E_C'] = credit_converter(row['L_E_C'])
        df.iloc[idx] = row

    # credit별 total 강의 수 체크
    print(df[['L_E_C', 'Rstu']].groupby(by=['L_E_C'], as_index=False).count())

    # 요일/시간별 수업/수강생 수 df화
    df = counter(df)
    df.to_csv('class_schedule.csv', encoding='utf-8-sig')
