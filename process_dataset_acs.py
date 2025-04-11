import numpy as np
import pandas as pd
import polars as pl
import os
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import shutil

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data/Info'
windowSize = 15

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default='de', help='Name of dataset. Options: de, nc')
parser.add_argument('--target', type=str, default='VEH', help='Name of target column')

args = parser.parse_args()

def fill_na_fix(df, na_method_list):
    """Fill na values in dataframe with the method defined in na_method_list."""
    for i in range(len(na_method_list)):
        na_process = na_method_list[i]
        column2process = df.columns[i]
        try:
            na_process = eval(na_process)
        except:
            pass

        if na_process == 'no' or na_process == 'pre':
            continue
        elif type(na_process) == int:
            df[column2process] = df[column2process].fillna(na_process)
        else:
            raise Exception("NaProcess is not defined correctly.")
        
    return df

def add_col_person(dfPerson: pl.DataFrame):
    """Add a column to each Variable in dfPerson that indicates whether it exists or not. This prepares for the window operation.
    """
    processingColumns = dfPerson.columns
    processedColumns = set()
    i = 0
    while i < len(processingColumns):
        columnHere = processingColumns[i]
        columnHereKey = columnHere.split('_')[0] if '_' in columnHere else columnHere
        if columnHere == 'SERIALNO':
            i += 1
            continue
        elif columnHereKey in processedColumns:
            i += 1
            continue
        else:
            personCols = dfPerson.columns
            notExistColName = f"{columnHereKey}_NotExist"
            dfPerson = dfPerson.with_columns(
                pl.lit(0).alias(notExistColName)
            )
            dfPerson = dfPerson.select(
                personCols[:i] + [notExistColName] + personCols[i:]
            )
            i += 1
            processedColumns.add(columnHereKey)
    
    return dfPerson

def combine(dfHouse, dfPerson, name, windowSize, varLenHouse, varLenPerson):
    """args: args.combinedMicroPath = combine_nc"""
    prepared_data_save_path = f'data/{name}_prepared/{name}_prepared.csv'
    prepared_data_save_path = Path(prepared_data_save_path).resolve()
    if prepared_data_save_path.is_file():
        print("Loading combined data from file...")
        data = pl.read_csv(prepared_data_save_path)
        newColumns = data.columns
        data = data.to_numpy().astype(np.float32)
    else:
        print("Combining data...")
        data = combineCore(dfHouse, dfPerson, windowSize, varLenHouse, varLenPerson)
        newColumns = []
        HouseColumns = dfHouse.columns
        HouseColumns.remove('SERIALNO')
        HouseColumns = [f"H_{i}" for i in HouseColumns]
        newColumns = HouseColumns
        PersonColumns = dfPerson.columns
        PersonColumns.remove('SERIALNO')
        for i in range(windowSize):
            PersonColumnsi = [f"P{i}_{j}" for j in PersonColumns]
            newColumns += PersonColumnsi
        print("Saving combined data to file...")
        prepared_data_save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data, columns=newColumns).to_csv(prepared_data_save_path, index=False)
        print("Done!")
    return data, newColumns

def combineCore(dfHouse, dfPerson, windowSize, varLenHouse, varLenPerson):
    """
    args:
        dfHouse: (4580, 98)
        dfPerson: (9092, 45)
        windowSize : 15
        varLenHouse: 97
        varLenPerson: 44
    return:
        data: (4580, 97+44*15)
    """
    data = np.zeros((len(dfHouse), varLenHouse + varLenPerson * windowSize), dtype=np.float32)
    SerialNoList = dfHouse['SERIALNO'].to_list()
    SerialNoList.sort()

    PersonColumns = dfPerson.columns
    PersonColumns.remove('SERIALNO')

    NotExistIndexList = []
    for i in range(len(PersonColumns)):
        if "NotExist" in PersonColumns[i]:
            NotExistIndexList.append(i)

    for i in tqdm(range(len(SerialNoList))):
        houseNodf = dfHouse.filter(pl.col('SERIALNO') == SerialNoList[i]).drop('SERIALNO')
        personNodf = dfPerson.filter(pl.col('SERIALNO') == SerialNoList[i]).drop('SERIALNO')
        placeholderPerson = np.zeros((windowSize, varLenPerson), dtype=np.float32)
        if len(personNodf) == 0:
            pass
        elif len(personNodf) > windowSize:
            personNodf = personNodf.limit(windowSize)
            placeholderPerson[:len(personNodf)] = personNodf.to_numpy()
        else:
            placeholderPerson[:len(personNodf)] = personNodf.to_numpy()
        
        placeholderPerson_dict = {}
        for idx_column in range(len(personNodf.columns)):
            placeholderPerson_dict[personNodf.columns[idx_column]] = placeholderPerson[:, idx_column]
        placeholderPerson = pl.DataFrame(placeholderPerson_dict)
        
        NotExistNameList = [PersonColumns[i] for i in NotExistIndexList]
        placeholderPerson = placeholderPerson.with_columns(pl.Series("index", range(len(placeholderPerson))))
        # Set the value of the column in NotExistIndexList to 1 if the index of placeholderPerson is greater than or equal to len(personNodf).
        placeholderPerson = placeholderPerson.with_columns([
            pl.when(pl.col("index") >= len(personNodf))
            .then(1)
            .otherwise(pl.col(col_name))
            # .keep_name()
            .alias(col_name)  # Use alias() instead of keep_name()
            for col_name in NotExistNameList
        ])
        placeholderPerson = placeholderPerson.drop("index")

        placeholderPerson = placeholderPerson.to_numpy()
        
        personHere = placeholderPerson.reshape(-1)  
        data[i] = np.concatenate((houseNodf.to_numpy().flatten(), personHere), axis=0)
    return data

# -------------------------------------------------------

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping

def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]


        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        print(train_df.columns)

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    

if __name__ == "__main__":
    name = args.dataname
    interest_df = pd.read_csv('InterestCsv.csv', header=0)

    target_col_name = args.target # Get target column name
    data_path = f'data/{name}_prepared/{name}_prepared.csv'
    
    info = {
        "name": f"{name}_prepared",
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": [], # num_columns,
        "cat_col_idx": [], # cat_columns,
        "target_col_idx": [], # target_columns
        "file_type": "csv",
        "data_path": data_path, # "data/{name}/{name}.csv"
        "test_path": None
    }

    if Path(data_path).exists():
        print(f"prepared data already exists at {data_path}, skip processing...")
        print(f"loading data from {data_path}")
        data_df = pd.read_csv(data_path)
    else:
        print("loading raw data and processing...")
        # 首先进行初始的预处理, 删除不需要的数据, 合并 household 和 person
        # data/acs_raw/psam_hde.csv
        household_path = f'data/acs_raw/psam_h{name}.csv'
        # data/acs_raw/psam_pde.csv
        person_path = f'data/acs_raw/psam_p{name}.csv'

        household_df = pd.read_csv(household_path)
        person_df = pd.read_csv(person_path)

        house_interest_list = interest_df[interest_df["house_or_indiv"] == "h"]["column_name"].tolist()
        indiv_interest_list = interest_df[interest_df["house_or_indiv"] == "i"]["column_name"].tolist()

        house_df = pd.read_csv(household_path, usecols=house_interest_list + ["ADJINC", "TYPEHUGQ", "NP"], dtype={1: str})
        indiv_df = pd.read_csv(person_path, usecols=indiv_interest_list, dtype={1: str})  

        ## 删除所有 TYPEHUGQ != 1 & NP == 0 的 record
        serialnos_to_delete = house_df[house_df["TYPEHUGQ"] != 1]['SERIALNO'].tolist()
        serialnos_to_delete += house_df[house_df["NP"] == 0]['SERIALNO'].tolist()

        house_df = house_df[~house_df['SERIALNO'].isin(serialnos_to_delete)]
        indiv_df = indiv_df[~indiv_df['SERIALNO'].isin(serialnos_to_delete)]

        house_df.reset_index(drop=True, inplace=True)
        indiv_df.reset_index(drop=True, inplace=True)

        ## 手动处理一些缺失值
        house_interest_df = interest_df.query("house_or_indiv=='h'")
        indiv_interest_df = interest_df.query("house_or_indiv=='i'")

        house_nan_list = house_interest_df["NaProcess"].tolist()
        indiv_nan_list = indiv_interest_df["NaProcess"].tolist()

        house_df = fill_na_fix(house_df, house_nan_list)
        indiv_df = fill_na_fix(indiv_df, indiv_nan_list)

        ## 手动处理一些特殊列
        # 1 HINCP: Household income (past 12 months, use ADJINC to adjust HINCP to  constant dollars)
        # 1 ADJINC: Adjustment factor for income and earnings dollar amounts (6 implied  decimal places) 1029928 .2021 factor (1.029928) 
        house_df["HINCP"] = house_df["HINCP"] * house_df["ADJINC"] / 1000000
        # 2 AGEP <25 set SCHL 0.
        SCHL_AGEP = indiv_df[["SCHL", "AGEP"]].values
        agep_less_than_25 = SCHL_AGEP[:, 1] < 25
        SCHL_AGEP[agep_less_than_25, 0] = 0
        indiv_df["SCHL"] = SCHL_AGEP[:, 0]

        house_df.drop(["ADJINC", "TYPEHUGQ"], axis=1, inplace=True)

        # 处理每一列的格式
        if "SERIALNO" in house_df.columns:
            house_df['SERIALNO'] = house_df['SERIALNO'].astype(str)
        if "TEN" in house_df.columns:
            house_df['TEN'] = house_df['TEN'].astype(int)
        if "VEH" in house_df.columns:
            house_df['VEH'] = house_df['VEH'].astype(int)
        if "HHL" in house_df.columns:
            house_df['HHL'] = house_df['HHL'].astype(int)
        if "HINCP" in house_df.columns:
            house_df['HINCP'] = house_df['HINCP'].astype(float)
        if "R18" in house_df.columns:
            house_df['R18'] = house_df['R18'].astype(int)
        if "R65" in house_df.columns:
            house_df['R65'] = house_df['R65'].astype(int)
        if "NP" in house_df.columns:
            house_df['NP'] = house_df['NP'].astype(int)

        if "SERIALNO" in indiv_df.columns:
            indiv_df['SERIALNO'] = indiv_df['SERIALNO'].astype(str)
        if "AGEP" in indiv_df.columns:
            indiv_df['AGEP'] = indiv_df['AGEP'].astype(int)
        if "SCHL" in indiv_df.columns:
            indiv_df['SCHL'] = indiv_df['SCHL'].astype(int)
        if "SEX" in indiv_df.columns:
            indiv_df['SEX'] = indiv_df['SEX'].astype(int)

        dfHouse = pl.from_pandas(house_df)
        dfPerson = pl.from_pandas(indiv_df)

        # Sort person data by age (AGEP), gender (SEX), and education level (SCHL)
        dfPerson = dfPerson.sort(["SERIALNO", "AGEP", "SEX", "SCHL"])

        # Add a column to each Variable in dfPerson that indicates whether it exists or not.
        dfPerson = add_col_person(dfPerson)
        
        # combine
        varLenHouse = len(dfHouse.columns) - 1 # 97 = 98 - SERIALNO
        varLenPerson = len(dfPerson.columns) - 1 # 52 = 53 - SERIALNO
        data_df, column_names = combine(dfHouse, dfPerson, name, windowSize, varLenHouse, varLenPerson)

        data_df = pd.DataFrame(data_df, columns=column_names)
        data_df = data_df.drop('H_NP', axis=1)
        data_df.to_csv(data_path, index=False)
    
    # # -------------------------------------------------------------
    # 再次按照 tabdiff 的格式处理数据
    print("Processing data again following tabdiff...")

    name = f"{name}_prepared"

    house_interest_type = interest_df[interest_df['house_or_indiv'] == 'h']
    indiv_interest_type = interest_df[interest_df['house_or_indiv'] == 'i']

    num_columns = []
    cat_columns = []
    
    type_df = deepcopy(house_interest_type)
    type_df = type_df.reset_index(drop=True)
    for i in range(len(type_df)):
        col_this = type_df.loc[i, "column_name"]
        if col_this == "SERIALNO":
            continue
        type_this = type_df.loc[i, "type"]

        if type_this == 'numeric':
            num_columns.append(f"H_{col_this}")
        elif type_this == 'categorical':
            cat_columns.append(f"H_{col_this}")
        else:
            raise ValueError(f"Invalid type: {type_this}")

    type_df = deepcopy(indiv_interest_type)
    type_df = type_df.reset_index(drop=True)
    for i in range(len(type_df)):
        col_this = type_df.loc[i, "column_name"]
        if col_this == "SERIALNO":
            continue
        type_this = type_df.loc[i, "type"]

        full_col_this = [f"P{i}_{col_this}" for i in range(windowSize)]
        full_col_notexist = [f"P{i}_{col_this}_NotExist" for i in range(windowSize)]

        if type_this == 'numeric':
            num_columns += full_col_this
        elif type_this == 'categorical':
            cat_columns += full_col_this
        else:
            raise ValueError(f"Invalid type: {type_this}")

        cat_columns += full_col_notexist


    # Define target column list
    target_columns = [target_col_name]

    # Remove target column from feature lists
    num_columns = [col for col in num_columns if target_col_name not in col]
    cat_columns = [col for col in cat_columns if target_col_name not in col]

    # Final column order: numerical features, categorical features, target column(s)
    column_names = num_columns + cat_columns + target_columns
    info['column_names'] = column_names # Store final column names in info

    num_data = data_df.shape[0]
    
    # Calculate indices based on the final column_names order
    info['num_col_idx'] = [column_names.index(col) for col in num_columns]
    info['cat_col_idx'] = [column_names.index(col) for col in cat_columns]
    info['target_col_idx'] = [column_names.index(col) for col in target_columns]

    # Use the final column order and calculated indices for mapping
    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, info['num_col_idx'], info['cat_col_idx'], info['target_col_idx'], column_names)

    name_idx_mapping = {val: key for key, val in idx_name_mapping.items()}

    int_columns = []
    int_col_idx = []
    int_col_idx_wrt_num = []
    for i, col_idx in enumerate(info['num_col_idx']):
        col = column_names[col_idx]
        col_data = data_df.iloc[:,col_idx]
        is_int = (col_data%1 == 0).all()
        if is_int:
            int_columns.append(col)
            int_col_idx.append(name_idx_mapping[col])
            int_col_idx_wrt_num.append(i)
    info['int_col_idx'] = int_col_idx
    info['int_columns'] = int_columns
    info['int_col_idx_wrt_num'] = int_col_idx_wrt_num   

    data_df.columns = range(len(data_df.columns))

    print(name, data_df.shape)

    col_info = {}
    
    for col_idx in info['num_col_idx']:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        data_df[col_idx] = data_df[col_idx].astype(float)
        col_info['max'] = float(data_df[col_idx].max())
        col_info['min'] = float(data_df[col_idx].min())
     
    for col_idx in info['cat_col_idx']:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        data_df[col_idx] = data_df[col_idx].astype(int)
        col_info['categorizes'] = list(set(data_df[col_idx]))    

    for col_idx in info['target_col_idx']:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            data_df[col_idx] = data_df[col_idx].astype(float)
            col_info['max'] = float(data_df[col_idx].max())
            col_info['min'] = float(data_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            data_df[col_idx] = data_df[col_idx].astype(int)
            col_info['categorizes'] = list(set(data_df[col_idx]))      

    info['column_info'] = col_info

    data_df.rename(columns = idx_name_mapping, inplace=True)

    if data_df.isna().any().any():
        raise ValueError("Training data contains nan in the numerical cols")

    X_num_train = data_df.iloc[:, info['num_col_idx']].to_numpy().astype(np.float32)
    X_cat_train = data_df.iloc[:, info['cat_col_idx']].to_numpy()
    y_train = data_df.iloc[:, info['target_col_idx']].to_numpy()
    
    test_df = data_df.sample(n=100, random_state=42)
    X_num_test = test_df.iloc[:, info['num_col_idx']].to_numpy().astype(np.float32)
    X_cat_test = test_df.iloc[:, info['cat_col_idx']].to_numpy()
    y_test = test_df.iloc[:, info['target_col_idx']].to_numpy()

    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    data_df.iloc[:, info['num_col_idx']] = data_df.iloc[:, info['num_col_idx']].astype(np.float32)
    data_df.to_csv(f'{save_dir}/train.csv', index = False)
    data_df.to_csv(f'{save_dir}/{name}.csv', index = False)

    info['data_path'] = f'{save_dir}/{name}.csv'

    test_df.iloc[:, info['num_col_idx']] = test_df.iloc[:, info['num_col_idx']].astype(np.float32)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
    
    data_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['train_num'] = data_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    # Add requested info
    info['name'] = name
    info['header'] = column_names # Assuming header refers to column names
    info['num_col_idx'] = info['num_col_idx']
    info['cat_col_idx'] = info['cat_col_idx']
    info['target_col_idx'] = info['target_col_idx']

    metadata = {'columns': {}}

    for i in info['num_col_idx']:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in info['cat_col_idx']:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'

    if info['task_type'] == 'regression':
        
        for i in info['target_col_idx']:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in info['target_col_idx']:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'
            
    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    # copy 一份到data/info
    shutil.copy(src=Path(f'{save_dir}/info.json').resolve(), dst=Path(f'data/Info/{name}.json').resolve())

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    cat = len(info['cat_col_idx'])
    num = len(info['num_col_idx'])
    print('Num', num)
    print('Int', len(info['int_col_idx']))
    print('Cat', cat)