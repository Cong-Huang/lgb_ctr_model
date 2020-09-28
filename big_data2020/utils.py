from math import radians, cos, sin, asin, sqrt 
import math 
import pandas as pd 
import numpy as np 
import datetime 


def standard(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)
    return x 

def pad_seq(arr, maxlen=1000):
    # [614, 10]
    if arr.shape[0] >= maxlen:
        return arr[:maxlen]
    else:
        return np.concatenate((arr, np.zeros((maxlen-arr.shape[0], arr.shape[1]))), axis=0)

def distance(lon1, lat1, lon2, lat2):
    return ((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2) ** 0.5

## 压缩内存 
def pandas_reduce_mem_usage(df):
    start_mem=df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    starttime = datetime.datetime.now()
    for col in df.columns:
        col_type=df[col].dtype   #每一列的类型
        if col_type !=object:    #不是object类型
            c_min=df[col].min()
            c_max=df[col].max()
            # print('{} column dtype is {} and begin convert to others'.format(col,col_type))
            if str(col_type)[:3]=='int':
                #是有符号整数
                if c_min<0:
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min >= np.iinfo(np.uint8).min and c_max<=np.iinfo(np.uint8).max:
                        df[col]=df[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.uint16).min and c_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.uint32).min and c_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
            #浮点数
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            # print('\t\tcolumn dtype is {}'.format(df[col].dtype))

        #是object类型，比如str
#         else:
#             # print('\t\tcolumns dtype is object and will convert to category')
#             df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    endtime = datetime.datetime.now()
    print('consume times: {:.4f}'.format((endtime - starttime).seconds))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df 


def rad(x):
    return x * math.pi / 180 

def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = rad(lon1), rad(lat1), rad(lon2), rad(lat2)
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(a ** 0.5) 
    r = 6371    # 地球平均半径，单位为公里
    return c * r 



def geohash_encode(longitude, latitude, precision=4):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)