=======================
Input data
=======================

Locations
=======================
|SP| input data is organized around the concept of a :term:`location`. Each location can have its own set of values for each of the input data fields or parameters.


Location hierarchy
=======================
Every location optionally has a parent location. The child location inherits all of the data field values
from the parent. The child location can override the values inherited from the parent.


Input parameters
=======================

location_name
     The name of the location. This needs to be the same as the name of the file, leaving off the ".json" suffix.

.. code-block:: python

    "Senegal", "Senegal-Dakar", "usa", "usa-Washington", "usa-Washington-seattle_metro"

data_provenance_notices
    A list of strings. Each string in the list describes the provenance of some portion, or all, of the
    data in the file.

.. code-block:: python

    ["This data originally comes from X, and co., 2015.", "Long term care facility (LTCF) data source is XYZ."]

reference_links
    A list of strings. Each string in the list is a reference for some portion, or all, of the data in the file.

.. code-block:: python

    ["https://github.com/awesomedata/awesome-public-datasets", "https://ingeniumcanada.org/collection-research/open-data"]

citations
    A list of strings. Each string in the list is a citation for some portion, or all, of the data in the file.

.. code-block:: python

    ["https://doi.org/10.1038/s41467-020-20544-y", "American Community Survey 2018: Seattle-Tacoma-Bellevue, WA"]

notes
    A list of strings. Each string in the list is a note describing something about the dataset.

.. code-block:: python

    ["Field X, row N, is missing from the source data and assumed to be default value Y.", "Data field Z is mising from the source data and assumed to have distribution A."]

parent
    The name of the parent location file, including the ".json" suffix.

.. code-block:: python

    "Senegal.json"

population_age_distribution_16

    The 16-bracket version of population age distribution. A list of tuples of the form [min_age, max_age, percentage].
    See next section for more info.

.. code-block:: python

    [
    [0, 4, 0.0605381173520116],
    [5, 9, 0.060734396722304],
    ...
    [70, 74, 0.0312168948061224],
    [75, 100, 0.0504085424578719]
    ]

population_age_distribution_18

    The 18-bracket version of population age distribution. A list of tuples of the form [min_age, max_age, percentage].
    See next section for more info.

.. code-block:: python

    [
    [0, 4, 0.0605381173520116],
    [5, 9, 0.060734396722304],
    ...
    [80, 84, 0.0140175336124184],
    [85, 100, 0.0166478127732105]
    ]

population_age_distribution_20

    The 20-bin version of population age distribution. A list of tuples of the form [min_age, max_age, percentage].
    See next section for more info.

.. code-block:: python

    [
    [0, 4, 0.0605381173520116],
    [5, 9, 0.060734396722304],
    ...
    [90, 94, 0.00436],
    [95, 100, 0.00236]
    ]

employment_rates_by_age

    Employment rate by age. A list of tuples of the form [age, percentage].

.. code-block:: python

    [
    [16, 0.3],
    ...
    [25, 0.861],
    ...
    [42, 0.838],
    ...
    [68, 0.294],
    ...
    [100, 0.061]
    ]

enrollment_rates_by_age

    School enrollment rate by age. A list of tuples of the form [age, percentage].

.. code-block:: python

    [
    ...
    [3, 0.529],
    ...
    [10, 0.987],
    ...
    [17, 0.977],
    ...
    [24, 0.409],
    ...
    [33, 0.113],
    ...
    [48, 0.027],
    ...
    [100, 0.0]
    ]

household_head_age_brackets

    Age brackets for the household head age distribution. A list of tuples of the form [age_min, age_max].

.. code-block:: python

    [
    [15, 19],
    [20, 24],
    [25, 29],
    [30, 34],
    [35, 39],
    [40, 44],
    [45, 49],
    [50, 54],
    [55, 59],
    [60, 64],
    [65, 69],
    [70, 74],
    [75, 79],
    [80, 100]
    ]

household_head_age_distribution_by_family_size

    A table providing the distribution of the age of the household head (sometimes referred to as the reference person), as a function of family size. Each row in this table specifies the distribution for a given family size. The family size is the first entry in the row.
    The remaining entries are, for each household head age bracket (see last table entry), the number or percentage of households with a household head in that age bracket.

.. code-block:: python

    [
    [1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [2, 163.0, 999.0, 2316.0, 2230.0, 1880.0, 1856.0, 2390.0, 3118.0, 9528.0, 9345.0, 5584.0],
    [3, 115.0, 757.0, 1545.0, 1907.0, 2066.0, 1811.0, 2028.0, 2175.0, 3311.0, 1587.0, 588.0],
    [4, 135.0, 442.0, 1029.0, 1951.0, 2670.0, 2547.0, 2368.0, 1695.0, 1763.0, 520.0, 221.0],
    [5, 61.0, 172.0, 394.0, 905.0, 1429.0, 1232.0, 969.0, 683.0, 623.0, 235.0, 94.0],
    [6, 25.0, 81.0, 153.0, 352.0, 511.0, 459.0, 372.0, 280.0, 280.0, 113.0, 49.0],
    [7, 24.0, 33.0, 63.0, 144.0, 279.0, 242.0, 219.0, 115.0, 157.0, 80.0, 16.0],
    [8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]

household_size_distribution

    Specifies the distribution of household sizes. A list of tuples of the form [household_size, percentage].

.. code-block:: python

    [
    [1, 0.2802338920169473],
    [2, 0.3425558454571084],
    [3, 0.154678770225653],
    [4, 0.1261686577488611],
    [5, 0.0589023321064863],
    [6, 0.0228368983653579],
    [7, 0.0146236040795857]
    ]

ltcf_resident_to_staff_ratio_distribution

    Specifies the distribution of the ratio of long term care facility residents to staff. A list of tuples of the form [ratio_low, ratio_high, percentage].

.. code-block:: python

    [
    ...
    [6.0, 6.0, 0.0227272727272727],
    ...
    [9.0, 9.0, 0.25],
    ...
    [14.0, 14.0, 0.0909090909090909]
    ]

ltcf_num_residents_distribution

    Specifies the distribution of number of long term care facility residents in a facility. A list of tuples of the form [num_low, num_high, percentage].

.. code-block:: python

    [
    ...
    [40.0, 59.0, 0.1343283582089552],
    ...
    [120.0, 139.0, 0.1194029850746268],
    ...
    [200.0, 219.0, 0.0149253731343283],
    ...
    [300.0, 319.0, 0.0298507462686567],
    ...
    [520.0, 539.0, 0.0149253731343283],
    ...
    [680.0, 699.0, 0.0]
    ]

ltcf_num_staff_distribution

Specifies the distribution of number of long term care facility staff in a facility. A list of tuples of the form [num_low, num_high, percentage].

.. code-block:: python

    [
    [0, 19,0.014925373134328358],
    ...
    [60, 79,0.1044776119402985],
    ...
    [140, 159,0.11940298507462686],
    ...
    [260, 279,0.04477611940298507],
    ...
    [460, 479,0.014925373134328358],
    ...
    [680, 699,0.0]
    ]

ltcf_use_rate_distribution

    Specifies the distribution of percentage of population of a given age that uses long term care facilities. A list of tuples of the form [age, percentage].

.. code-block:: python

    [
    ...
    [57.0, 0.0],
    ...
    [63.0, 0.01014726],
    ...
    [72.0, 0.00992606],
    ...
    [84.0, 0.06078108],
    ...
    [91.0, 0.18420189],
    ...
    [100.0, 0.18420189]
    ]

school_size_brackets

    Specifies the school size (number of students) brackets associated with the school size distribution data. A list of tuples of the form [school_size_low, school_size_hi].

.. code-block:: python

    [
    [20, 50],
    [51, 100],
    [101, 300],
    [301, 500],
    [501, 700],
    [701, 900],
    [901, 1100],
    [1101, 1300],
    [1301, 1500],
    [1501, 1700],
    [1701, 1900],
    [1901, 2100],
    [2101, 2300],
    [2301, 2700]
    ]

school_size_distribution

    Specifies the percentage of schools for each school_size_bracket (see last table entry). A list of percentages, one for each entry in school_size_brackets.

.. code-block:: python

    [0.02752293577981651, 0.009174311926605502, 0.20183486238532117, 0.39449541284403683, 0.19266055045871566, 0.045871559633027505, 0.05504587155963302, 0.036697247706422007, 0.009174311926605502, 0.0, 0.02752293577981651, 0.0, 0.0, 0.0]

school_size_distribution_by_type

    Specifies the percentage of schools for each school_size_bracket, broken out by school type. A list of json objects with two keys 'school_type', and 'size_distribution'. The 'school_type' entry is a string. The 'size_distribution' entry is a list of percentages, one for each entry in school_size_brackets.

.. code-block:: python

    [{
    "school_type": "ms",
    "size_distribution": [0.0, 0.0, 0.0, 0.0, 0.4166666666666667, 0.16666666666666666, 0.3333333333333333, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }, {
    "school_type": "hs",
    "size_distribution": [0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.0, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.2, 0.06666666666666667, 0.0, 0.2, 0.0, 0.0, 0.0]
    }, {
    "school_type": "uv",
    "size_distribution": [0.10720338983050849, 0.06059322033898306, 0.15974576271186441, 0.27796610169491537, 0.22754237288135598, 0.07754237288135594, 0.024152542372881364, 0.016525423728813562, 0.013135593220338982, 0.013135593220338982, 0.01016949152542373, 0.006355932203389832, 0.0046610169491525435, 0.0012711864406779662]
    }, {
    "school_type": "pk",
    "size_distribution": [0.0, 0.0, 0.22580645161290322, 0.6129032258064516, 0.16129032258064516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }, {
    "school_type": "es",
    "size_distribution": [0.0, 0.0, 0.22580645161290322, 0.6129032258064516, 0.16129032258064516, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }]

school_types_by_age

    Specifies the age ranges for each school type.

.. code-block:: python

    [{
    "school_type": "pk",
    "age_range": [3, 5]
    }, {
    "school_type": "es",
    "age_range": [6, 10]
    }, {
    "school_type": "ms",
    "age_range": [11, 13]
    }, {
    "school_type": "hs",
    "age_range": [14, 17]
    }, {
    "school_type": "uv",
    "age_range": [18, 100]
    }]

workplace_size_counts_by_num_personnel

    Specifies the count of workplaces broken down by number of workplace personnel.

.. code-block:: python

    [
    [1, 4, 60050.0],
    [5, 9, 19002.0],
    [10, 19, 13625.0],
    [20, 49, 9462.0],
    [50, 99, 3190.0],
    [100, 249, 1802.0],
    [250, 499, 486.0],
    [500, 999, 157.0],
    [1000, 1999, 109.0]
    ]

16-, 18-, and 20-bracket versions of population age distributions.
==================================================================
The are different aggregations of the age distribution for a population for a variety of reasons.
These kind of data come from sources like a national census website or survey sample and may be 
aggregated into age brackets (also referred to as groups or bins), or may be available for single years of age. The age brackets are also used to map other data such as age-specific contact matrices. Contact matrices of age mixing patterns are rarely available at a resolution of single years of age. Rather, they are most frequently
available for age brackets. Currently, by default, |SP| uses age-specific contact matrices aggregated to 16 age brackets and so we include the age distributions of locations aggregated to 16 age brackets, as well as other aggregations. 

Specifically, for US sourced data we include the original US Census Bureau age distributions aggregated to 18 age brackets, and age distributions inferred for 20 age brackets from trend data to assist in infectious disease modeling of older ages. Where inferred or estimated, we include a note in the 'notes' field about the method used to infer or estimate the age distribution data.


Location File Format
=======================

.. code-block:: python

   todo


Example Input File
=======================

.. code-block:: python

   todo
