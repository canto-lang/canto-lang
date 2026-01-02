/**
 * Mutual Exclusivity Detection for Conflict Analysis
 *
 * These predicates detect when rule conditions are mutually exclusive,
 * meaning they cannot both be satisfied simultaneously. This is crucial
 * for accurate conflict detection - rules with mutually exclusive conditions
 * should NOT be flagged as conflicts.
 *
 * Example: If rule1 uses is_like(X, fruit_terms) and rule2 uses
 * is_like(X, vehicle_terms), these are mutually exclusive because
 * an input cannot match both categories simultaneously.
 *
 * This file is consulted by delp_meta.pl and shares its dynamic predicates:
 * - rule_info/4
 * - sup/2
 */

%% ============================================================================
%% Category Extraction
%% ============================================================================

/**
 * condition_category(+Condition, -Variable, -Category)
 * Extract the variable and category from a category-based condition.
 * Fails if the condition doesn't reference a category.
 */
condition_category(is_like(Var, Cat), Var, Cat).
condition_category(has_any_like(Var, Cat), Var, Cat).
condition_category(has_all_like(Var, Cat), Var, Cat).
condition_category(has_none_like(Var, Cat), Var, Cat).
condition_category(length_where_like_gt(Var, _, Cat, _), Var, Cat).
condition_category(length_where_like_eq(Var, _, Cat, _), Var, Cat).
condition_category(length_where_like_lt(Var, _, Cat, _), Var, Cat).
condition_category(length_where_like_gte(Var, _, Cat, _), Var, Cat).
condition_category(length_where_like_lte(Var, _, Cat, _), Var, Cat).
% Handle negation - recurse into inner condition
condition_category(\+(Cond), Var, Cat) :-
    condition_category(Cond, Var, Cat).
condition_category(\\+(Cond), Var, Cat) :-
    condition_category(Cond, Var, Cat).

/**
 * extract_body_categories(+Body, -Categories)
 * Extract ALL categories from a rule body as a list of Var-Cat pairs.
 * Handles De Morgan OR: \+ ((\+(A)), (\+(B))) -> extracts from A and B
 */
extract_body_categories(Body, Categories) :-
    findall(Var-Cat,
            (member(Cond, Body), extract_condition_category(Cond, Var, Cat)),
            Categories).

/**
 * extract_condition_category(+Condition, -Variable, -Category)
 * Extract category from a condition, handling De Morgan OR transformation.
 * OR is represented as: \+ ((\+(A)), (\+(B))) meaning A OR B
 */
% Simple category conditions
extract_condition_category(is_like(Var, Cat), Var, Cat).
extract_condition_category(has_any_like(Var, Cat), Var, Cat).
extract_condition_category(has_all_like(Var, Cat), Var, Cat).
extract_condition_category(has_none_like(Var, Cat), Var, Cat).
extract_condition_category(length_where_like_gt(Var, _, Cat, _), Var, Cat).
extract_condition_category(length_where_like_eq(Var, _, Cat, _), Var, Cat).
extract_condition_category(length_where_like_lt(Var, _, Cat, _), Var, Cat).
extract_condition_category(length_where_like_gte(Var, _, Cat, _), Var, Cat).
extract_condition_category(length_where_like_lte(Var, _, Cat, _), Var, Cat).

% Handle negation - first try direct extraction from inner condition
% This handles \+(has_any_like(...)) which is semantically "has none like"
extract_condition_category(\+(Inner), Var, Cat) :-
    (   extract_condition_category(Inner, Var, Cat)  % Simple negation of category condition
    ;   extract_from_demorgan_or(Inner, Var, Cat)    % De Morgan OR structure
    ).
extract_condition_category(\\+(Inner), Var, Cat) :-
    (   extract_condition_category(Inner, Var, Cat)
    ;   extract_from_demorgan_or(Inner, Var, Cat)
    ).

/**
 * extract_from_demorgan_or(+Conjunction, -Variable, -Category)
 * Extract categories from De Morgan OR structure: ((\+(A)), (\+(B)), ...)
 * Handles nested De Morgan structures from chained ORs (A OR B OR C).
 */
% Conjunction of negations - extract from each branch
extract_from_demorgan_or((A, B), Var, Cat) :-
    (   extract_from_demorgan_or(A, Var, Cat)
    ;   extract_from_demorgan_or(B, Var, Cat)
    ).
% Nested De Morgan OR: \+(\+ (...)) is from chained ORs - recurse into inner
extract_from_demorgan_or(\+(\+(Inner)), Var, Cat) :-
    !,
    extract_from_demorgan_or(Inner, Var, Cat).
extract_from_demorgan_or(\\+(\\+(Inner)), Var, Cat) :-
    !,
    extract_from_demorgan_or(Inner, Var, Cat).
extract_from_demorgan_or(\+(\\+(Inner)), Var, Cat) :-
    !,
    extract_from_demorgan_or(Inner, Var, Cat).
extract_from_demorgan_or(\\+(\+(Inner)), Var, Cat) :-
    !,
    extract_from_demorgan_or(Inner, Var, Cat).
% Single negated condition - extract from inner
extract_from_demorgan_or(\+(Inner), Var, Cat) :-
    extract_condition_category(Inner, Var, Cat).
extract_from_demorgan_or(\\+(Inner), Var, Cat) :-
    extract_condition_category(Inner, Var, Cat).

%% ============================================================================
%% Mutual Exclusivity Detection
%% ============================================================================

/**
 * bodies_mutually_exclusive(+Body1, +Body2)
 * True if two rule bodies CANNOT both be satisfied simultaneously.
 * This happens when they have category-based conditions on the same variable
 * but with different categories and NO overlapping categories.
 *
 * For OR conditions (De Morgan form), we check if there's ANY overlap
 * between the category sets. If no overlap exists, they're mutually exclusive.
 *
 * IMPORTANT: If an OR has a branch without category conditions (e.g., flag is true),
 * we CANNOT prove mutual exclusivity because that branch could overlap with anything.
 *
 * Conservative behavior: If we can't prove exclusivity, assume potential conflict.
 */
bodies_mutually_exclusive(Body1, Body2) :-
    % Check for quantifier implications first (e.g., has_all vs not(has_any))
    quantifier_implications_exclusive(Body1, Body2),
    !.
bodies_mutually_exclusive(Body1, Body2) :-
    % Both bodies must be "category-complete" (all OR branches have categories)
    body_is_category_complete(Body1),
    body_is_category_complete(Body2),
    % Extract all categories from both bodies (handles De Morgan OR)
    extract_body_categories(Body1, Cats1),
    extract_body_categories(Body2, Cats2),
    % Both must have category conditions
    Cats1 \= [],
    Cats2 \= [],
    % Check if categories are exclusive (no overlap on same variable)
    categories_are_exclusive(Cats1, Cats2),
    !.

%% ============================================================================
%% Quantifier Implication Exclusivity
%% ============================================================================

/**
 * quantifier_implications_exclusive(+Body1, +Body2)
 * True if bodies are mutually exclusive due to quantifier implications:
 * - has_all_like(L, C) implies has_any_like(L, C)
 *   So: has_all_like(L, C) and \+(has_any_like(L, C)) are exclusive
 * - has_none_like(L, C) implies \+(has_any_like(L, C))
 *   So: has_none_like(L, C) and has_any_like(L, C) are exclusive
 * - has_all_like(L, C) implies \+(has_none_like(L, C))
 *   So: has_all_like(L, C) and has_none_like(L, C) are exclusive
 */
quantifier_implications_exclusive(Body1, Body2) :-
    member(Cond1, Body1),
    member(Cond2, Body2),
    conditions_exclusive_by_quantifier(Cond1, Cond2),
    !.

/**
 * conditions_exclusive_by_quantifier(+Cond1, +Cond2)
 * True if two conditions are mutually exclusive due to quantifier logic.
 */
% has_all_like implies has_any_like, so has_all vs not(has_any) are exclusive
conditions_exclusive_by_quantifier(has_all_like(L, C), \+(has_any_like(L, C))).
conditions_exclusive_by_quantifier(\+(has_any_like(L, C)), has_all_like(L, C)).
conditions_exclusive_by_quantifier(has_all_like(L, C), \\+(has_any_like(L, C))).
conditions_exclusive_by_quantifier(\\+(has_any_like(L, C)), has_all_like(L, C)).

% has_none_like is equivalent to not(has_any_like), so has_none vs has_any are exclusive
conditions_exclusive_by_quantifier(has_none_like(L, C), has_any_like(L, C)).
conditions_exclusive_by_quantifier(has_any_like(L, C), has_none_like(L, C)).

% not(has_any_like) and has_any_like are obviously exclusive
conditions_exclusive_by_quantifier(\+(has_any_like(L, C)), has_any_like(L, C)).
conditions_exclusive_by_quantifier(has_any_like(L, C), \+(has_any_like(L, C))).
conditions_exclusive_by_quantifier(\\+(has_any_like(L, C)), has_any_like(L, C)).
conditions_exclusive_by_quantifier(has_any_like(L, C), \\+(has_any_like(L, C))).

% has_all and has_none are mutually exclusive (can't have all matching AND none matching)
conditions_exclusive_by_quantifier(has_all_like(L, C), has_none_like(L, C)).
conditions_exclusive_by_quantifier(has_none_like(L, C), has_all_like(L, C)).

% Same patterns for eq variants
conditions_exclusive_by_quantifier(has_all_eq(L, V), \+(has_any_eq(L, V))).
conditions_exclusive_by_quantifier(\+(has_any_eq(L, V)), has_all_eq(L, V)).
conditions_exclusive_by_quantifier(has_all_eq(L, V), \\+(has_any_eq(L, V))).
conditions_exclusive_by_quantifier(\\+(has_any_eq(L, V)), has_all_eq(L, V)).
conditions_exclusive_by_quantifier(has_none_eq(L, V), has_any_eq(L, V)).
conditions_exclusive_by_quantifier(has_any_eq(L, V), has_none_eq(L, V)).
conditions_exclusive_by_quantifier(has_all_eq(L, V), has_none_eq(L, V)).
conditions_exclusive_by_quantifier(has_none_eq(L, V), has_all_eq(L, V)).
conditions_exclusive_by_quantifier(\+(has_any_eq(L, V)), has_any_eq(L, V)).
conditions_exclusive_by_quantifier(has_any_eq(L, V), \+(has_any_eq(L, V))).
conditions_exclusive_by_quantifier(\\+(has_any_eq(L, V)), has_any_eq(L, V)).
conditions_exclusive_by_quantifier(has_any_eq(L, V), \\+(has_any_eq(L, V))).

% Same patterns for neq variants
conditions_exclusive_by_quantifier(has_all_neq(L, V), \+(has_any_neq(L, V))).
conditions_exclusive_by_quantifier(\+(has_any_neq(L, V)), has_all_neq(L, V)).
conditions_exclusive_by_quantifier(has_all_neq(L, V), \\+(has_any_neq(L, V))).
conditions_exclusive_by_quantifier(\\+(has_any_neq(L, V)), has_all_neq(L, V)).
conditions_exclusive_by_quantifier(has_none_neq(L, V), has_any_neq(L, V)).
conditions_exclusive_by_quantifier(has_any_neq(L, V), has_none_neq(L, V)).
conditions_exclusive_by_quantifier(has_all_neq(L, V), has_none_neq(L, V)).
conditions_exclusive_by_quantifier(has_none_neq(L, V), has_all_neq(L, V)).
conditions_exclusive_by_quantifier(\+(has_any_neq(L, V)), has_any_neq(L, V)).
conditions_exclusive_by_quantifier(has_any_neq(L, V), \+(has_any_neq(L, V))).
conditions_exclusive_by_quantifier(\\+(has_any_neq(L, V)), has_any_neq(L, V)).
conditions_exclusive_by_quantifier(has_any_neq(L, V), \\+(has_any_neq(L, V))).

/**
 * body_is_category_complete(+Body)
 * True if all conditions in the body have category information.
 * For OR conditions (De Morgan form), all branches must have categories.
 * If any branch lacks a category, we can't prove mutual exclusivity.
 */
body_is_category_complete(Body) :-
    \+ (member(Cond, Body), condition_has_non_category_branch(Cond)).

/**
 * condition_has_non_category_branch(+Condition)
 * True if the condition contains a branch without category information.
 * This includes:
 * - De Morgan OR with a branch that has no category
 * - Conditions like flag(true) that don't involve categories
 */
% De Morgan OR - check all branches
condition_has_non_category_branch(\+(Inner)) :-
    demorgan_has_non_category_branch(Inner).
condition_has_non_category_branch(\\+(Inner)) :-
    demorgan_has_non_category_branch(Inner).

/**
 * demorgan_has_non_category_branch(+Conjunction)
 * Check if any branch of a De Morgan OR lacks category information.
 * Handles nested De Morgan structures from chained ORs (A OR B OR C).
 */
% Conjunction - check both branches
demorgan_has_non_category_branch((A, B)) :-
    (   demorgan_has_non_category_branch(A)
    ;   demorgan_has_non_category_branch(B)
    ).
% Nested De Morgan OR: \+(\+ (...)) is from chained ORs - recurse into inner
demorgan_has_non_category_branch(\+(\+(Inner))) :-
    !,
    demorgan_has_non_category_branch(Inner).
demorgan_has_non_category_branch(\\+(\\+(Inner))) :-
    !,
    demorgan_has_non_category_branch(Inner).
demorgan_has_non_category_branch(\+(\\+(Inner))) :-
    !,
    demorgan_has_non_category_branch(Inner).
demorgan_has_non_category_branch(\\+(\+(Inner))) :-
    !,
    demorgan_has_non_category_branch(Inner).
% Single negated condition - check if it's NOT a category condition
demorgan_has_non_category_branch(\+(Inner)) :-
    \+ is_category_condition(Inner).
demorgan_has_non_category_branch(\\+(Inner)) :-
    \+ is_category_condition(Inner).

/**
 * is_category_condition(+Condition)
 * True if the condition is a category-based condition.
 */
is_category_condition(is_like(_, _)).
is_category_condition(has_any_like(_, _)).
is_category_condition(has_all_like(_, _)).
is_category_condition(has_none_like(_, _)).
is_category_condition(length_where_like_gt(_, _, _, _)).
is_category_condition(length_where_like_eq(_, _, _, _)).
is_category_condition(length_where_like_lt(_, _, _, _)).
is_category_condition(length_where_like_gte(_, _, _, _)).
is_category_condition(length_where_like_lte(_, _, _, _)).

/**
 * categories_are_exclusive(+Cats1, +Cats2)
 * True if two category lists have NO overlap on the same variable.
 *
 * For mutual exclusivity:
 * - If both bodies check the same variable, the categories must be different
 * - If ANY category from Cats1 matches ANY category from Cats2, they can overlap
 *
 * Example:
 * - Cats1 = [input-cat_a, input-cat_b], Cats2 = [input-cat_c, input-cat_d]
 *   -> Exclusive (no shared categories)
 * - Cats1 = [input-cat_a, input-cat_b], Cats2 = [input-cat_b, input-cat_c]
 *   -> NOT exclusive (cat_b appears in both)
 */
categories_are_exclusive(Cats1, Cats2) :-
    % Find categories for the same variable in both lists
    member(Var-_, Cats1),
    member(Var-_, Cats2),
    !,
    % Get all categories for this variable from each list
    findall(Cat, member(Var-Cat, Cats1), CatList1),
    findall(Cat, member(Var-Cat, Cats2), CatList2),
    % Check that no category appears in both lists
    \+ (member(C, CatList1), member(C, CatList2)).

%% ============================================================================
%% Conflict Analysis (requires rule_info/4 and sup/2 from delp_meta.pl)
%% ============================================================================

/**
 * variable_has_real_conflict(+Variable)
 * True if at least two rules for Variable can both fire simultaneously
 * (i.e., their bodies are NOT mutually exclusive) and produce different values.
 */
variable_has_real_conflict(Variable) :-
    % Find two rules for this variable with different values
    rule_info(RuleId1, Goal1, _, Body1),
    rule_info(RuleId2, Goal2, _, Body2),
    Goal1 =.. [Variable, Value1],
    Goal2 =.. [Variable, Value2],
    RuleId1 @< RuleId2,              % Avoid checking same pair twice
    Value1 \= Value2,                 % Different conclusions
    \+ bodies_mutually_exclusive(Body1, Body2),  % Can coexist
    !.

/**
 * get_variable_conflicts(-Report)
 * Analyze all variables and return conflict information.
 * This is the main entry point for conflict analysis from Python.
 *
 * Returns a dict with:
 *   - conflicts: List of variables with real conflicts
 *   - no_conflicts: List of variables without conflicts
 */
get_variable_conflicts(Report) :-
    findall(Var, (rule_info(_, Goal, _, _), Goal =.. [Var|_]), AllVars),
    sort(AllVars, Variables),
    analyze_all_variable_conflicts(Variables, Conflicts, NoConflicts),
    Report = _{
        conflicts: Conflicts,
        no_conflicts: NoConflicts
    }.

/**
 * analyze_all_variable_conflicts(+Variables, -Conflicts, -NoConflicts)
 * Categorize variables into those with conflicts and those without.
 */
analyze_all_variable_conflicts([], [], []).
analyze_all_variable_conflicts([Var|Rest], Conflicts, NoConflicts) :-
    analyze_all_variable_conflicts(Rest, RestConflicts, RestNoConflicts),
    analyze_variable_conflict_status(Var, VarInfo),
    (   VarInfo.has_conflict = true ->
        Conflicts = [VarInfo|RestConflicts],
        NoConflicts = RestNoConflicts
    ;   Conflicts = RestConflicts,
        NoConflicts = [VarInfo|RestNoConflicts]
    ).

/**
 * analyze_variable_conflict_status(+Variable, -Analysis)
 * Analyze a single variable for conflict status.
 */
analyze_variable_conflict_status(Variable, Analysis) :-
    % Get all rules and values for this variable
    findall(
        rule_data(RuleId, ValueAtom, Type),
        (   rule_info(RuleId, Goal, Type, _Body),
            Goal =.. [Variable, Value],
            safe_value_to_atom(Value, ValueAtom)
        ),
        RuleDataList
    ),
    length(RuleDataList, NumRules),

    % Get unique values
    findall(V, member(rule_data(_, V, _), RuleDataList), Values),
    sort(Values, UniqueValues),
    length(UniqueValues, NumValues),

    % Determine if there's a real conflict using mutual exclusivity check
    (   NumRules > 1, NumValues > 1, variable_has_real_conflict(Variable) ->
        HasConflict = true,
        analyze_rule_conflict_statuses(Variable, RuleDataList, RulesWithStatus)
    ;   HasConflict = false,
        maplist(rule_data_to_default_dict, RuleDataList, RulesWithStatus)
    ),

    Analysis = _{
        variable: Variable,
        num_rules: NumRules,
        num_values: NumValues,
        has_conflict: HasConflict,
        rules: RulesWithStatus
    }.

/**
 * safe_value_to_atom(+Value, -Atom)
 * Convert a Prolog value to a safe atom for serialization.
 */
safe_value_to_atom(Value, Atom) :-
    (   atom(Value) -> Atom = Value
    ;   number(Value) -> atom_number(Atom, Value)
    ;   string(Value) -> atom_string(Atom, Value)
    ;   term_to_atom(Value, Atom)
    ).

/**
 * rule_data_to_default_dict(+RuleData, -Dict)
 * Convert rule_data to dict with default winning status.
 */
rule_data_to_default_dict(rule_data(RuleId, Value, Type), Dict) :-
    Dict = _{
        rule_id: RuleId,
        value: Value,
        type: Type,
        status: wins,
        defeated_by: [],
        has_override: false
    }.

/**
 * analyze_rule_conflict_statuses(+Variable, +RuleDataList, -RulesWithStatus)
 * For each rule, determine if it wins or is defeated.
 */
analyze_rule_conflict_statuses(_, [], []).
analyze_rule_conflict_statuses(Variable, [rule_data(RuleId, Value, Type)|Rest], [RuleDict|RestDicts]) :-
    % Check if this rule has "overriding all"
    (   sup(RuleId, _) -> HasOverride = true ; HasOverride = false ),

    % Find rules that defeat this one
    findall(
        _{rule_id: OtherRuleId, value: OtherValueAtom},
        (   rule_info(OtherRuleId, OtherGoal, _, _),
            OtherRuleId \= RuleId,
            OtherGoal =.. [Variable, OtherValue],
            safe_value_to_atom(OtherValue, OtherValueAtom),
            OtherValueAtom \= Value,
            sup(OtherRuleId, RuleId)
        ),
        DefeatedBy
    ),

    % Determine status
    (   DefeatedBy = [] -> Status = wins ; Status = defeated ),

    RuleDict = _{
        rule_id: RuleId,
        value: Value,
        type: Type,
        status: Status,
        defeated_by: DefeatedBy,
        has_override: HasOverride
    },

    analyze_rule_conflict_statuses(Variable, Rest, RestDicts).
