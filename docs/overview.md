
# `synthpops` overview

The role of `synthpops` is to construct synthetic networks of people that satisfy statistical properties of real-world populations (such as the age distribution, household size, etc.). These synthetic populations can then be used in agent based models like `covasim` to simulate epidemics.

Fundamentally, the network can be considered a multilayer network where

- Nodes are people, with attributes like age
- Edges represent interactions between people, with attributes like the setting in which the interactions take place (e.g. 'household','school', 'work'). The relationship between the interaction setting and properties governing disease transmission (e.g. frequency of contact, risk associated with each contact) is mapped separately by `covasim`. Synthpops reports whether the edge exists or not. Note that the relevant quantity in `covasim` is the parameter `beta`, which captures the probability of transmission via a given edge per timestep. The value of this parameter captures both number of effective contacts for disease transmission and transmission probability per contact. 

The generated network is a multilayer network in the sense that it is possible for people to be connected by multiple edges each in different layers of the network. The layers are referred to as _contact layers_. For example, the 'work' contact layer is a representation of all of the pairwise connections between people at work, and the 'household' contact layer represents the pairwise connections between household members. Typically these networks are clustered e.g. everyone within a household interacts with each other, but not with other households (they may interact with members of households instead via their school, work or as community contacts). 

`synthpops` functions in two stages

1. Generate people living in households, and then assign individuals to workplaces and schools. Save the output to a cache file on disk. Implemented in `synthpops.generate_synthetic_population`. 
2. Load the cached file and produce a dictionary that can be used by `covasim`. Implemented in `synthpops.make_population`. `covasim` assigns community contacts at random on a daily basis to reflect the random and stochastic aspect of contacts in many public spaces, such as shopping centres, parks, and community centres.  

## Algorithm


This section describes the algorithm used to generate the connections between people in each contact layer for a given location in the real world. The fundamental algorithm is the same for homes, schools, and workplaces, but with some variations for each.

The method draws upon previously published models to infer high resolution age specific contact patterns in different physical settings and locations (Mossong et al. 2008, Fumanelli et al. 2012, Prem et al. 2017, Mistry et al. 2020). The general idea is to use age-specific contact matrices which describe age mixing patterns for a specific population. Here, we use the contact matrices from "Projecting social contact matrices in 152 countries using contact surveys and demographic data" by Prem et al. (2017), however other age-specific contact matrices can be used with `synthpops`. The matrices represent the average number of contacts between people for different age bins (the default matrices use 5 year age bins). For example, a household of two individuals is relatively unlikely to consist of a 25 year old and a 15 year old, so for the 25-29 year age bin, there are a low number of expected contacts with the 15-19 year age bin (c.f., Fig. 2c in Prem et al.). 

The overall workflow is contained in `synthpops.generate_synthetic_population` and is described below. We start by generating our population in the household layer.
1. A collection of households is instantiated with sizes drawn from census data.
2. For each household, the age of a 'reference' person is sampled from data on that maps household size to a reference person in those households. The reference person may be referred to as the head of the household, a parent in the household, or some other definition specific to the data being used. If no data mapping household size to ages of reference individuals are available, then the age of the reference person is sampled from the age distribution of adults for the location. 
3. The age bin of the reference people indentifies the row of the contact matrix for that location. The remaining household members are then selected by sampling an age for the distribution of contacts for the reference person's age (i.e., normalizing the values of the row and sampling for a column) and assigning someone with that age to the household.
4. As households are generated, individuals are given ids. 
5. With households constructed, students are chosen according to enrollment data by age.
6. Students are assigned to schools using a similar method as above, where we select the age of a reference person and then select their contacts in school from age specific contact matrix for the school setting and data on school sizes.
7. With all students assigned to schools, teachers are selected from the labor force according to employment data. 
8. The rest of the labor force are assigned to workplaces by selecting a reference person and their contacts using an age specific contact matrix and data on workplace sizes.

We now go through the specific data sources and setting-specific particulars

## Instantiating people

(Population generated through households, not a pool of people. This is why we may generate a population that doesn't match the age distribution well. This and the next section on households are in sort of a reverse order. First, use the age profile given in age brackets to produce a an age distribution in 1 year age bins. Then start creating households based on size and attributes of reference person. Then fill in household members and this produces the age distribution.) 

The first step in producing a synthetic population is to construct the pool of people (ages and sexes). The core implementation is in `synthpops.get_age_sex_n` and this is called via `synthpops.get_usa_age_sex_n` which additionally loads data files for the USA from disk.

## Instantiating households

The first step in producing a synthetic population is to construct households with individuals of different ages living together. The core implementation is in `synthpops.generate_all_households` which is called in `synthpops.generate_synthetic_population`. 
The inputs required for this step are:

**Age bracket distribution** specifying the distribution of people in age bins for the location. For example:

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
	75_100       , 0.05110673396642193

**Gender fraction by age bracket** specifying the proportion of people of each sex in each age bin (if specifying gender) (Note to us: maybe we remove this because it's not being used)

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
	75_100       , 0.40409019781651095 , 0.5959098021834891

First, a single or 1-year age distribution is generated from the binned age distribution. This is done using `sp.get_age_n` to create a samples of ages from the binned distribution, and then normalizing to create a single-year distribution. This distribution can therefore be gathered using whatever age bins are present in any given dataset.

## Household contacts

As locations, households are special in that

- Unlike schools and workplaces, everyone must be assigned to a household
- The size of the household is important (e.g., a 2 person household looks very different in comparison to a 5 or 6 person household) and some households only have 1 person.
- The reference person/head of the household can be well defined by data

Three data sets are required for households:

**Age distribution of the reference person for each household size** - the distribution is what matters so it doesn't matter if absolute counts are available or not, each *row* is normalized. If this is not available, default to sampling the age of the reference individual from the age distribution for adults.

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

By default, `synthpops` uses matrices from a study (Prem et al. 2017) that projected inferred age mixing patterns from the POLYMOD study (Mossong et al. 2008) in Europe to other countries. `synthpops` can take in user specified contact matrices if other age mixing patterns are available for the household, school, and workplace settings (see Zenodo's Social Contact Patterns page for other empirical contact matrices from survey studies). 

In theory the household contact matrix varies with household size, but in general data at that resolution is unavailable. 

The `synthpops` functions implement household initialization as follows

1. `synthpops.generate_household_sizes_from_fixed_pop_size` generates empty households with known size based on the distribution of household sizes
2. `synthpops.generate_all_households` takes in the remaining sources above, and then
	- Calls `generate_living_alone` to populate households with 1 person (either from data on those living alone, or if unavailable, from the adult age distribution)
	- Calls `generate_larger_households` repeatedly with with different household sizes to populate those households, first sampling the age of a reference person and then their household contacts as outlined above.

## Schools

Schools are special in that

- Enrollment rates by age determine the probability of individual being a student given their age
- Staff members such as teachers are chosen from individuals determined to be in the adult labor force
- The current methods in `synthpops` treat student and worker status as mutually exclusive. Many young adults may be both students and workers, part time or full time in either status. The ability to select individuals to participate in both activities will be introduced in a later develop of the model.

The data required for schools are:

**School size distribution**

	school_size , percent
	0-50        , 0.2
	51-100      , 0.1
	101-300     , 0.3

**Enrollment by age** specifying the percentage of people of each age attending school. See `synthpops.get_school_enrollment_rates`: 
but note that this mainly implements parsing a Seattle-specific data file to produce the following data structure, which could equivalently be read directly from a file:

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

**School contact matrix** specifying the number/weight of contacts by age bin. Similar to the household contact matrix. For example:

	        0-10        , 10-20       , 20-30
	0-10    0.659867911 , 0.503965302 , 0.214772978
	10-20   0.314776879 , 0.895460015 , 0.412465791
	20-30   0.132821425 , 0.405073038 , 1.433888594

**Employment rates by age** which is used when determining who is in the labor force, and thus which adults are available to be chosen as teachers

	Age , Percent
	16  , 0.496
	17  , 0.496
	18  , 0.496
	19  , 0.496
	20  , 0.838
	21  , 0.838
	22  , 0.838

**Student teacher ratio** which the average ratio for the location. Methods to use a distribution or ratio for different types of schools may come in later developments of the model.

	student_teacher_ratio=30

Typically, contact matrices describing age-specific mixing patterns in schools include the interactions between students and their teachers. These patterns describe multiple types of schools, from possibly preschools to universities. 

The workflow is then

1. `synthpops.get_uids_in_school` uses the enrolment rates to determine which people attend school. This then provides the number of students needing to be assigned to schools.
2. `synthpops.generate_school_sizes` generates schools according to the school size distribution until there are enough places for every student to be assigned a school
3. `synthpops.send_students_to_school` assigns specific students to specific schools
	- This function is similar to households in that a reference student is selected, and then the contact matrix is used to fill the remaining spots in the school(all of the following is drawn from metholody l(all of the following is drawn from metholody learned from past papers so reference them here)(all of the following is drawn from metholody learned from past papers so reference them here)earned from past papers so reference them here)
	- Some particulars in this function deal with ensuring a teacher/adult is less likely to be selected as a reference person, and restricting the age range of sampled people relative to the reference person so that a primary school age reference person will result in the rest of the school being populated with other primary school age children
4. Next, teachers are selected by first getting a pool of working age people that are not students. This is implemented in `synthpops.get_uids_potential_workers`. This population is then filtered further by employment rates in `get_workers_by_age_to_assign` resulting in a collection of people that need to be assigned workplaces.
5. In `assign_teachers_to_work`, for each school, work out how many teachers are needed according to the number of students and the student-teacher ratio, and sample those teachers from the pool of adult workers. A minimum and maximum age for teachers can be provided to select teachers from a specified range of ages (this can be used to account for the additional years of education needed to become a teacher in many places).

## Workplaces

_Again, note that work and school are currently exclusive, because the people attending schools are removed from the list of eligible workers. This doesn't necessarily need to be the case though_ (in fact, we know that in any countries and cultures around the world, people take on multiple roles as both students and workers, either part-time or full-time in one or both activities).

Finally, all remaining workers are assigned to workplaces. Workplaces are special in that there is little/no age structure so workers of all ages may be present in every workplace.

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

**Work contact matrix** specifying the number/weight of contacts by age bin. Similar to the household contact matrix. For example:

	        20-30       , 30-40       , 40-50
	20-30   0.659867911 , 0.503965302 , 0.214772978
	30-40   0.314776879 , 0.895460015 , 0.412465791
	40-50   0.132821425 , 0.405073038 , 1.433888594

The workflow is

1. `synthpops.generate_workplace_sizes` generates workplace sizes according to the workplace size distribution until the number of workers is reached
2. `synthpops.assign_rest_of_workers` populates workplaces just like for households and schools - randomly selecting the age of a reference person, and then sampling the rest of the workplace using the contact matrix. 
