import synthpops


dropbox_path = os.path.join('/home','dmistry','Dropbox (IDM)','seattle_network')
census_location = 'seattle_metro'


age_bracket_distr = read_age_bracket_distr(dropbox_path,census_location)

gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,census_location)

age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
age_brackets = get_age_brackets_from_df(age_brackets_filepath)
age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)


a,s = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)
print(a,s)