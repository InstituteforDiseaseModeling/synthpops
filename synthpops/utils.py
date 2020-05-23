import os
import platform
import pandas as pd


def get_kc_snf_df_location():

    os_platform = platform.system()
    userpath = os.path.expanduser('~')
    username = os.path.split(userpath)[-1]
    if os_platform == "Windows":
        KC_snf_df = pd.read_csv(
            os.path.join('C:/Users', username, 'Dropbox (IDM)', 'COVID-19', 'seattle_network', 'secure_King_County',
                         'IDM_CASE_FACILITY.csv'))
    elif os_platform != "Windows":
        KC_snf_df = pd.read_csv(
            os.path.join('/home', username, 'Dropbox (IDM)', 'COVID-19', 'seattle_network', 'secure_King_County',
                         'IDM_CASE_FACILITY.csv'))

    return KC_snf_df
