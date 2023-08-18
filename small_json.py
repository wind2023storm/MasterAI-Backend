import pandas as pd
import requests
import json

def make_get(url, headers=None, params=None):
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        return response.json()
    else:
        return 400


def fetch_and_store_items_recursive(url, headers) -> pd.DataFrame:
    """
    return a dataframe with columns id, name
    """
    number = 0
    file_number = 0
    cursor = None
    categories = []
    while True:
        params = None
        if cursor:
            params = {
                'cursor': cursor,
            }

        json_data = make_get(url, headers, params)
        if json_data != 400:
            for i in json_data['receipts']:
                number = number + 1
                categories.append(i)
                if number > 49999:
                    number = 0
                    file_number = file_number + 1 
                    with open(f'receipts{file_number}.json', "w") as file:
                        json.dump(categories, file)
                    categories = []
                else:
                    print(number)
        else:
            break
        
        cursor = json_data.get('cursor', None)
        if not cursor:
            break
    print("Extration End")
    file_number = file_number + 1 
    with open(f'receipts{file_number}.json', "w") as file:
        json.dump(categories, file)

    # return pd.DataFrame(
    #     categories,
    #     columns=['receipt_number', 'source', 'total_money']
    # )


if __name__ == '__main__':

    token = '603e2570ff4b4c73ac993d5515b24e24'

    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = 'https://api.loyverse.com/v1.0/receipts?&limit=250#'

    fetch_and_store_items_recursive(url, headers)