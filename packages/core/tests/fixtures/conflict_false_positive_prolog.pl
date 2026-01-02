% ============ DYNAMIC PREDICATES ============
:- dynamic is_like/2.
:- dynamic has/2.
:- dynamic has_property/3.
:- dynamic rule_info/4.
:- dynamic sup/2.
:- dynamic pattern/2.
:- dynamic has_any_like/2.
:- dynamic has_any_eq/2.
:- dynamic has_any_neq/2.
:- dynamic has_all_like/2.
:- dynamic has_all_eq/2.
:- dynamic has_all_neq/2.
:- dynamic has_none_like/2.
:- dynamic has_none_eq/2.
:- dynamic has_none_neq/2.
:- dynamic has_any_prop_eq/4.
:- dynamic has_any_prop_like/4.
:- dynamic has_all_prop_eq/4.
:- dynamic has_all_prop_like/4.
:- dynamic length_where_is_eq/4.
:- dynamic length_where_is_gt/4.
:- dynamic length_where_is_lt/4.
:- dynamic length_where_is_gte/4.
:- dynamic length_where_is_lte/4.
:- dynamic length_where_like_eq/4.
:- dynamic length_where_like_gt/4.
:- dynamic length_where_like_lt/4.
:- dynamic length_where_like_gte/4.
:- dynamic length_where_like_lte/4.
:- dynamic let_any/4.
:- dynamic let_all/4.
:- dynamic let_none/4.
:- dynamic compare_gt/2.
:- dynamic compare_lt/2.
:- dynamic compare_gte/2.
:- dynamic compare_lte/2.
:- dynamic input/1.
:- dynamic result/1.
:- discontiguous result/1.

% ============ DECLARATIONS ============
% Variable: input
%   the user input
% Variable: result

% ============ SEMANTIC CATEGORIES ============
% Category: fruit_terms
pattern(fruit_terms, apple).
pattern(fruit_terms, orange).
pattern(fruit_terms, banana).
pattern(fruit_terms, grape).
% Category: vehicle_terms
pattern(vehicle_terms, car).
pattern(vehicle_terms, truck).
pattern(vehicle_terms, bike).
pattern(vehicle_terms, bus).

% ============ HAS RELATIONSHIPS ============

% ============ FACTS ============

% ============ STRICT RULES ============
% Rule: result_1
result('fruit') :- is_like(input, fruit_terms).
% Rule: result_2
result('vehicle') :- is_like(input, vehicle_terms).

% ============ DEFEASIBLE RULES ============
% Note: Using standard Prolog (:-)  - defeasibility handled via superiority

% ============ RULE METADATA (for DeLP meta-interpreter) ============
% rule_info(RuleId, Head, Type, BodyList)
rule_info(result_1, result('fruit'), strict, [is_like(input, fruit_terms)]).
rule_info(result_2, result('vehicle'), strict, [is_like(input, vehicle_terms)]).

% Generated predicates for: fruit_terms
% Description: N/A
% Generated via LLM (ILP-inspired)

pattern(fruit_terms, apple).
pattern(fruit_terms, orange).
pattern(fruit_terms, banana).
pattern(fruit_terms, grape).
pattern(fruit_terms, pear).
pattern(fruit_terms, peach).
pattern(fruit_terms, plum).
pattern(fruit_terms, apricot).
pattern(fruit_terms, nectarine).
pattern(fruit_terms, cherry).
pattern(fruit_terms, strawberry).
pattern(fruit_terms, raspberry).
pattern(fruit_terms, blueberry).
pattern(fruit_terms, blackberry).
pattern(fruit_terms, cranberry).
pattern(fruit_terms, boysenberry).
pattern(fruit_terms, gooseberry).
pattern(fruit_terms, kiwi).
pattern(fruit_terms, kiwifruit).
pattern(fruit_terms, mango).
pattern(fruit_terms, papaya).
pattern(fruit_terms, pineapple).
pattern(fruit_terms, melon).
pattern(fruit_terms, watermelon).
pattern(fruit_terms, cantaloupe).
pattern(fruit_terms, honeydew).
pattern(fruit_terms, lemon).
pattern(fruit_terms, lime).
pattern(fruit_terms, grapefruit).
pattern(fruit_terms, tangerine).
pattern(fruit_terms, clementine).
pattern(fruit_terms, mandarin).
pattern(fruit_terms, satsuma).
pattern(fruit_terms, pomelo).
pattern(fruit_terms, kumquat).
pattern(fruit_terms, pomegranate).
pattern(fruit_terms, fig).
pattern(fruit_terms, date).
pattern(fruit_terms, coconut).
pattern(fruit_terms, avocado).
pattern(fruit_terms, guava).
pattern(fruit_terms, lychee).
pattern(fruit_terms, litchi).
pattern(fruit_terms, rambutan).
pattern(fruit_terms, durian).
pattern(fruit_terms, jackfruit).
pattern(fruit_terms, passionfruit).
pattern(fruit_terms, persimmon).
pattern(fruit_terms, dragonfruit).
pattern(fruit_terms, pitaya).
pattern(fruit_terms, starfruit).
pattern(fruit_terms, carambola).
pattern(fruit_terms, plantain).
pattern(fruit_terms, currant).
pattern(fruit_terms, redcurrant).
pattern(fruit_terms, blackcurrant).
pattern(fruit_terms, elderberry).
pattern(fruit_terms, mulberry).
pattern(fruit_terms, loganberry).
pattern(fruit_terms, cloudberry).
pattern(fruit_terms, ackee).
pattern(fruit_terms, soursop).
pattern(fruit_terms, cherimoya).
pattern(fruit_terms, tamarind).
pattern(fruit_terms, loquat).
pattern(fruit_terms, quince).
pattern(fruit_terms, ugli).
pattern(fruit_terms, feijoa).
pattern(fruit_terms, longan).
pattern(fruit_terms, jabuticaba).
pattern(fruit_terms, medlar).
pattern(fruit_terms, crabapple).
pattern(fruit_terms, blood_orange).
pattern(fruit_terms, navel_orange).
pattern(fruit_terms, ananas).
pattern(fruit_terms, mirabelle).
pattern(fruit_terms, kumera_fruit).
pattern(fruit_terms, key_lime).
pattern(fruit_terms, finger_lime).
pattern(fruit_terms, boysen_berry).
pattern(fruit_terms, sugar_apple).
pattern(fruit_terms, custard_apple).
pattern(fruit_terms, rose_apple).
pattern(fruit_terms, star_apple).
pattern(fruit_terms, breadfruit).
pattern(fruit_terms, langsat).
pattern(fruit_terms, mangosteen).
pattern(fruit_terms, sapodilla).
pattern(fruit_terms, horned_melon).
pattern(fruit_terms, kiwano).
pattern(fruit_terms, pepino).
pattern(fruit_terms, acerola).
pattern(fruit_terms, barberry).
pattern(fruit_terms, sea_buckthorn).
pattern(fruit_terms, goji_berry).
pattern(fruit_terms, huckleberry).
pattern(fruit_terms, serviceberry).
pattern(fruit_terms, rowanberry).
pattern(fruit_terms, rosehip).
pattern(fruit_terms, jujube).
pattern(fruit_terms, yuzu).
pattern(fruit_terms, calamansi).

% Generated predicates for: vehicle_terms
% Description: N/A
% Generated via LLM (ILP-inspired)

pattern(vehicle_terms, car).
pattern(vehicle_terms, auto).
pattern(vehicle_terms, automobile).
pattern(vehicle_terms, truck).
pattern(vehicle_terms, lorry).
pattern(vehicle_terms, pickup_truck).
pattern(vehicle_terms, pickup).
pattern(vehicle_terms, bike).
pattern(vehicle_terms, bicycle).
pattern(vehicle_terms, motorbike).
pattern(vehicle_terms, motorcycle).
pattern(vehicle_terms, scooter).
pattern(vehicle_terms, moped).
pattern(vehicle_terms, bus).
pattern(vehicle_terms, minibus).
pattern(vehicle_terms, coach).
pattern(vehicle_terms, van).
pattern(vehicle_terms, minivan).
pattern(vehicle_terms, suv).
pattern(vehicle_terms, crossover).
pattern(vehicle_terms, jeep).
pattern(vehicle_terms, taxi).
pattern(vehicle_terms, cab).
pattern(vehicle_terms, uber).
pattern(vehicle_terms, rideshare).
pattern(vehicle_terms, tram).
pattern(vehicle_terms, streetcar).
pattern(vehicle_terms, train).
pattern(vehicle_terms, subway).
pattern(vehicle_terms, metro).
pattern(vehicle_terms, monorail).
pattern(vehicle_terms, light_rail).
pattern(vehicle_terms, ferry).
pattern(vehicle_terms, boat).
pattern(vehicle_terms, ship).
pattern(vehicle_terms, yacht).
pattern(vehicle_terms, sailboat).
pattern(vehicle_terms, canoe).
pattern(vehicle_terms, kayak).
pattern(vehicle_terms, aircraft).
pattern(vehicle_terms, airplane).
pattern(vehicle_terms, aeroplane).
pattern(vehicle_terms, plane).
pattern(vehicle_terms, jet).
pattern(vehicle_terms, helicopter).
pattern(vehicle_terms, glider).
pattern(vehicle_terms, spacecraft).
pattern(vehicle_terms, spaceship).
pattern(vehicle_terms, rocket).
pattern(vehicle_terms, scooter_electric).
pattern(vehicle_terms, e_bike).
pattern(vehicle_terms, electric_bike).
pattern(vehicle_terms, electric_car).
pattern(vehicle_terms, hybrid_car).
pattern(vehicle_terms, motorhome).
pattern(vehicle_terms, campervan).
pattern(vehicle_terms, rv).
pattern(vehicle_terms, caravan).
pattern(vehicle_terms, trailer).
pattern(vehicle_terms, semi_truck).
pattern(vehicle_terms, articulated_lorry).
pattern(vehicle_terms, bus_coach).
pattern(vehicle_terms, school_bus).
pattern(vehicle_terms, city_bus).
pattern(vehicle_terms, tanker_truck).
pattern(vehicle_terms, fire_truck).
pattern(vehicle_terms, fire_engine).
pattern(vehicle_terms, ambulance).
pattern(vehicle_terms, police_car).
pattern(vehicle_terms, patrol_car).
pattern(vehicle_terms, squad_car).
pattern(vehicle_terms, snowmobile).
pattern(vehicle_terms, atv).
pattern(vehicle_terms, quad_bike).
pattern(vehicle_terms, golf_cart).
pattern(vehicle_terms, forklift).
pattern(vehicle_terms, bulldozer).
pattern(vehicle_terms, excavator).
pattern(vehicle_terms, crane_truck).
pattern(vehicle_terms, delivery_van).
pattern(vehicle_terms, cargo_van).
pattern(vehicle_terms, box_truck).
pattern(vehicle_terms, handcart).
pattern(vehicle_terms, cart).
pattern(vehicle_terms, rickshaw).
pattern(vehicle_terms, tuk_tuk).
pattern(vehicle_terms, scooter_mobility).
pattern(vehicle_terms, wheelchair).
pattern(vehicle_terms, skateboard).
pattern(vehicle_terms, longboard).
pattern(vehicle_terms, hoverboard).
pattern(vehicle_terms, segway).
pattern(vehicle_terms, vehicle).
pattern(vehicle_terms, motor_vehicle).
pattern(vehicle_terms, road_vehicle).
pattern(vehicle_terms, watercraft).
pattern(vehicle_terms, seaplane).
pattern(vehicle_terms, hovercraft).
pattern(vehicle_terms, gondola).
pattern(vehicle_terms, taxi_boat).
pattern(vehicle_terms, air_taxi).
pattern(vehicle_terms, bike_share).
pattern(vehicle_terms, car_share).
pattern(vehicle_terms, carpool).
pattern(vehicle_terms, ridesharing).
pattern(vehicle_terms, auto_rickshaw).
pattern(vehicle_terms, autobus).
pattern(vehicle_terms, autocar).
pattern(vehicle_terms, minibus_shuttle).
pattern(vehicle_terms, shuttle_bus).