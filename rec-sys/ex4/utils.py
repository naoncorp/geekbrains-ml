import numpy as np
import pandas as pd


def prefilter_items(data: pd.DataFrame,
                    group_col: str=None,
                    popular_col: str=None,
                    time_col: str=None,
                    price_col: str=None,
                    top_popular_filter: int = None,
                    top_unpopular_filter: int = None,
                    time_unpopular_filter: int = None,
                    chip_item_filter: int = None,
                    exp_item_filter: int = None) -> pd.DataFrame:

    n_filter_start = (data['item_id'] == 999999).sum()
    
    # фильтр самых популярных товаров
    if top_popular_filter:  

        popularity = data.groupby(group_col)[popular_col].sum().reset_index()
        
        top = popularity.sort_values(popular_col, ascending=False).head(top_popular_filter).item_id.tolist()
        
        data.loc[data[group_col].isin(top), group_col] = 999999
    
    # фильтр самых популярных товаров
    if top_unpopular_filter:  

        bottom = popularity.sort_values(popular_col, ascending=True).head(top_unpopular_filter).item_id.tolist()
        
        data.loc[data[group_col].isin(bottom), group_col] = 999999
    
    # фильтр товаров, которые не продавались за последние N месяцев
    if time_unpopular_filter:  

        actuality = data.groupby(group_col)[time_col].min().reset_index()
        
        top_actual = actuality[actuality[time_col] > 365].item_id.tolist()
        
        data.loc[data[group_col].isin(top_actual), group_col] = 999999
    
    # Фильт товаров, которые стоят < N$
    if chip_item_filter:  

        low_price = data[data[price_col] < chip_item_filter].item_id.tolist()
        
        data.loc[data[group_col].isin(low_price), group_col] = 999999
    
    # Фильт товаров, которые стоят > N$ (дорогих)
    if exp_item_filter:  

        high_price = data[data[price_col] > exp_item_filter].item_id.tolist()
        
        data.loc[data[group_col].isin(high_price), group_col] = 999999

    n_filter = (data['item_id'] == 999999).sum() - n_filter_start
  

    return data
    