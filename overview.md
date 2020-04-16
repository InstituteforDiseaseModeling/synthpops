# `synthpops` overview

The role of `synthpops` is to construct random networks of people that satisfy statistical properties of specific real-world populations (such as age distribution, household size). These synthetic populations can then be used in agent based models like `covasim` to simulate epidemics.

Fundamentally, the network can be considered a multigraph where

- Nodes are people, with attributes like age and sex
- Edges represent interactions between people, with attributes like type (e.g. 'school', 'work'). The relationship between the type of interaction and properties governing disease transmission (e.g. frequency of contact, risk associated with each contact) is mapped separately by `covasim`. Synthpops simply reports whether the edge exists or not. Note that the relevant quantiy in `covasim` is the parameter `beta`, which captures the probability of transmission via a given edge per timestep. The value of this parameter captures both number of contacts and transmission probability per contact. 

The network is a multigraph in the sense that it is possible for people to be connected multiple types by edges of different types. The types of edges are referred to as _contact layers_. For example, the 'work' contact layer is a representation of all of the pairwise connections between people associated with work, and the 'household' contact layer represents the pairwise connections associated with homes. Typically these networks are clustered e.g. everyone within a household interacts with each other, but not with any other households (which they may interact with instead via work or random community contacts). 

`synthpops` functions in two stages

1. Generate households, workplaces, schools, people, and assign people to each location. Save the output to a cache file on disk. Implemented in `synthpops.generate_synthetic_population`
2. Load the cached file and produce a dictionary that can be used by `covasim`. Implemented in `synthpops.make_contacts`

# Algorithm

This section describes the algorithm used to determine the interactions between people in each contact layer. The fundamental algorithm is the same for homes, schools, and workplaces, but with some minor variations for each.

The general idea is to use the contact matrices (age mixing matrices) in "Projecting social contact matrices in 152 countries using contact surveys and demographic data" by Prem et al. (2017). These matrices represent the average number of contacts between people in 10 year age bins. For example, a household is relatively unlikely consist of a 25 year old and a 15 year old, so for the 25-30 year age bin, there are a low number of expected contacts with the 15-20 year age bin (c.f., Fig. 2c in Prem et al.). 

The overall workflow is contained in `synthpops.generate_synthetic_population` and is described below. We start with a population of people - that is, a pool of people that have already been assigned ages and sexes. For each location

1. A collection of locations (homes, schools, workplaces) is instantiated with sizes drawn from a setting-specific distribution
2. For each location, a 'reference person' is randomly selected and added to the location. 
3. The age bin of the reference person identifies the row of the contact matrix for that location. The remaining people at the location are then selected by sampling an age for the distribution of contacts for the reference person (i.e., sampling from a column of the contact matrix) and assigning someone with that age to that location

We now go through the specific data sources and setting-specific particulars

## Instantiating people

The first step in producing a synthetic population is to construct the pool of people (ages and sexes). The core implementation is in `synthpops.get_age_sex_n` and this is called via `synthpops.get_usa_age_sex_n` which additionally loads data files for the USA from disk. 

The inputs required for this step are:

**Age bracket distribution** specifying the proportion of people in age bins. For example:

	age_bracket , percent
	0_4         , 0.0594714358950416
	5_9         , 0.06031137308234759
	10_14       , 0.05338015778985113
	15_19       , 0.054500690394160285
	20_24       , 0.06161403846144956
	25_29       , 0.08899312471888453
	30_34       , 0.0883533486774803
	35_39       , 0.07780767611060545
	40_44       , 0.07099017823587304
	45_49       , 0.06996903280562596
	50_54       , 0.06655242534751997
	55_59       , 0.06350008343899961
	60_64       , 0.05761405140489549
	65_69       , 0.04487122889235999
	70_74       , 0.030964420778483555
	75_79       , 0.05110673396642193

**Gender fraction by age bracket** specifying the proportion of people of each sex in each age bin

	age_bracket , fraction_male       , fraction_female
	0_4         , 0.5099026832074123  , 0.4900973167925878
	5_9         , 0.5088948332554093  , 0.49110516674459065
	10_14       , 0.5107858843905607  , 0.4892141156094393
	15_19       , 0.5085878590997379  , 0.4914121409002621
	20_24       , 0.5099071927494805  , 0.49009280725051957
	25_29       , 0.5172428122483962  , 0.48275718775160376
	30_34       , 0.5162427863611868  , 0.4837572136388132
	35_39       , 0.5191185731769626  , 0.48088142682303736
	40_44       , 0.504929348179983   , 0.4950706518200169
	45_49       , 0.5102437219627248  , 0.4897562780372751
	50_54       , 0.5051538514968397  , 0.4948461485031604
	55_59       , 0.5011465635851405  , 0.49885343641485946
	60_64       , 0.48506827992361634 , 0.5149317200763837
	65_69       , 0.4738430790785841  , 0.5261569209214159
	70_74       , 0.4619759942672877  , 0.5380240057327123
	75_99       , 0.40409019781651095 , 0.5959098021834891

People are sampled using 1-year age bins (so they are assigned an actual age) and therefore these distributions are 'interpolated' into single year distributions. This is done in several different places but fundamentally performs the same thing - normalizing all data sources into single-year distributions. These data can therefore be gathered using whatever age bins are present in any given dataset.

The output of this population generation process is a collection of ages and sexes for each person in a population of a given size.

## Household contacts

As locations, households are special in that

- Unlike schools and workplaces, everyone must be assigned to a household
- The exact size of the household is important (e.g., a 2 person household looks very different to a 4 person household) and some households only have 1 person
- The head of the household/reference person can be well defined in data

Three data sets are required for households:

**Age of the head of the household for each household size** - the distribution is what matters so it doesn't matter if absolute counts are available or not, each *row* could be normalized.

	family_size , 18-20 , 20-24 , 25-29 , 30-34 , 35-39 , 40-44 , 45-49 , 50-54 , 55-64 , 65-74 , 75-99
	2           , 163   , 999   , 2316  , 2230  , 1880  , 1856  , 2390  , 3118  , 9528  , 9345  , 5584
	3           , 115   , 757   , 1545  , 1907  , 2066  , 1811  , 2028  , 2175  , 3311  , 1587  , 588
	4           , 135   , 442   , 1029  , 1951  , 2670  , 2547  , 2368  , 1695  , 1763  , 520   , 221
	5           , 61    , 172   , 394   , 905   , 1429  , 1232  , 969   , 683   , 623   , 235   , 94
	6           , 25    , 81    , 153   , 352   , 511   , 459   , 372   , 280   , 280   , 113   , 49
	7           , 24    , 33    , 63    , 144   , 279   , 242   , 219   , 115   , 157   , 80    , 16

**Distribution of household sizes**

	household_size , percent
	1              , 0.2781590909877753
	2              , 0.3443313103056699
	3              , 0.15759535523004006
	4              , 0.13654311541644018
	5              , 0.050887858718118274
	6              , 0.019738368167953997
	7              , 0.012744901174002305

**Household contact matrix** specifying the number/weight of contacts by age bin. For example

	        0-10        , 10-20       , 20-30
	0-10    0.659867911 , 0.503965302 , 0.214772978
	10-20   0.314776879 , 0.895460015 , 0.412465791
	20-30   0.132821425 , 0.405073038 , 1.433888594

In the first instance, these matrices can be drawn from the Prem et al. paper, although other data sources may be preferable because that paper generalizes European social structures to other countries which may not always be appropriate. 

In theory this household contact matrix should vary with household size, but it is unlikely that data at that resolution is available. 

The `synthpops` functions implement household initialization as follows

1. `synthpops.generate_household_sizes_from_fixed_pop_size` generates empty households with known size based on the distribution of household sizes
2. `synthpops.generate_all_households` takes in the remaining sources above, and then
	- Calls `generate_living_alone` to populate households with 1 person
	- Calls `generate_larger_households` repeatedly with with different household sizes to populate those households

## Schools

Schools are special in that

- Only school-age children are eligible to go to school
- Some people work as teachers at schools

The data required for schools are:

**School size distribution**

	school_size , percent
	0-50        , 0.2
	51-100      , 0.1
	101-300     , 0.3

**Enrolment by age** specifying the percentage of people of each age attending school. See `synthpops.get_school_enrollment_rates` but note that this mainly implements parsing a Seattle-specific data file to produce the following data structure, which could equivalently be read directly from a file:

	age , percent
	0   , 0
	1   , 0
	2   , 0
	3   , 0.529
	4   , 0.529
	5   , 0.95
	6   , 0.95
	7   , 0.95
	8   , 0.95
	9   , 0.95
	10  , 0.987
	11  , 0.987
	12  , 0.987
	13  , 0.987

**School contact matrix** specifying the number/weight of contacts by age bin. Similar to the household contact matrix and also provided by the Prem et al. paper. For example:

	        0-10        , 10-20       , 20-30
	0-10    0.659867911 , 0.503965302 , 0.214772978
	10-20   0.314776879 , 0.895460015 , 0.412465791
	20-30   0.132821425 , 0.405073038 , 1.433888594

**Employment rates by age** which is used when selecting teachers

	Age , Percent
	16  , 0.496
	17  , 0.496
	18  , 0.496
	19  , 0.496
	20  , 0.838
	21  , 0.838
	22  , 0.838

**Student teacher ratio** which is just a single number for the setting.

	student_teacher_ratio=30

One particular is that the matrices in the Prem et al. paper include interactions between students and teachers. Also, they include universities.

The workflow is then

1. `synthpops.get_uids_in_school` uses the enrolment rates to determine which people attend school. This then provides the number of students needing to be assigned to schools.
2. `synthpops.generate_school_sizes` generates schools according to the school size distribution until there are enough places for every students to be assigned a school
3. `synthpops.send_students_to_school` assigns specific students to specific schools
	- This function is similar to households in that a reference person is selected, and then the contact matrix is used to fill the remaining spots in the school
	- Some particulars in this function deal with ensuring a teacher/adult is less likely to be selected as a reference person, and restricting the age range of sampled people relative to the reference person so that a primary school age reference person will result in the rest of the school being populated with other primary school age children
4. Next, teachers are selected by first getting a pool of working age people that are not at school. This is implemented in `synthpops.get_uids_potential_workers`. This population is then filtered further by employment rates in `get_workers_by_age_to_assign` resulting in a collection of people that need to be assigned workplaces.
5. In `assign_teachers_to_work`, for each school, work out how many teachers are there, and sample those teachers from the pool of workers

## Workplaces

_Note that work and school are currently exclusive, because the people attending schools are removed from the list of eligible workers. This doesn't necessarily need to be the case though_

Finally, all remaining workers are assigned to workplaces. Workplaces are special in that there is little/no age structure so workers of all ages are present in every workplace.

The data required for workplaces are:

**Workplace size distribution** - again, this gets normalized so can be specified as absolute counts or as normalized values.

	work_size_bracket , size_count
	1-4               , 2947
	5-9               , 992
	10-19             , 639
	20-49             , 430
	50-99             , 140
	100-249           , 83
	250-499           , 26
	500-999           , 13
	1000-1999         , 12

**Work contact matrix** specifying the number/weight of contacts by age bin. Similar to the household contact matrix and also provided by the Prem et al. paper. For example:

	        20-30       , 30-40       , 40-50
	20-30   0.659867911 , 0.503965302 , 0.214772978
	30-40   0.314776879 , 0.895460015 , 0.412465791
	40-50   0.132821425 , 0.405073038 , 1.433888594

The workflow is

1. `synthpops.generate_workplace_sizes` generates workplaces according to the workplace size distribution until the number of workers is reached
2. `synthpops.assign_rest_of_workers` populates workplaces just like for households and schools - randomly selecting a reference person, and then sampling the rest of the workplace using the contact matrix. 
