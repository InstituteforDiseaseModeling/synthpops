import synthpops as sp


if __name__ == '__main__':
    sp.validate()
    datadir = sp.settings_config.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 2000
    plot = True
    # plot = False
    use_default = False

    sp.generate_synthetic_population(n,
                                     datadir,
                                     location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     sheet_name=sheet_name,
                                     plot=plot, use_default=use_default)
