/**
 * DeLP Static Analyzer
 * Compile-time validation and static analysis of DeLP programs
 *
 * Validates:
 * - No circular superiority relations
 * - No transitive cycles in superiority
 * - No contradictory strict rules without resolution
 * - All superiority references exist
 * - Proper rule structure
 * - Unreachable/always-defeated rules
 * - Semantic consistency
 *
 * Usage:
 *   validate_delp_program(Errors).
 *   Errors: List of validation error terms
 */

:- dynamic validation_error/2.
:- dynamic rule_info/4.
:- dynamic sup/2.

%% ============================================================================
%% Main Validation Entry Point
%% ============================================================================

/**
 * validate_delp_program(-Errors)
 * Run all validation checks and collect errors
 * Returns empty list if valid, list of error terms otherwise
 */
validate_delp_program(Errors) :-
    % Clear previous errors
    retractall(validation_error(_, _)),

    % Run all validation checks (order matters for some checks)
    validate_rule_structure,
    validate_superiority_references,
    validate_no_circular_superiority,
    validate_no_transitive_cycles,
    validate_no_contradictory_strict_rules,
    validate_no_unreachable_rules,
    validate_semantic_consistency,
    validate_rule_body_references,

    % Collect all errors as dicts for Janus, converting terms to strings
    findall(_{type: Type, details: StringDetails}, (
        validation_error(Type, Details),
        term_string(Details, StringDetails)
    ), Errors).

/**
 * validate_delp_program_or_fail
 * Validate and fail with error message if invalid
 */
validate_delp_program_or_fail :-
    validate_delp_program(Errors),
    (   Errors = [] ->
        format('✓ DeLP program validation passed~n'),
        true
    ;   format('✗ DeLP program validation FAILED:~n'),
        print_validation_errors(Errors),
        fail
    ).

%% ============================================================================
%% Validation: Circular Superiority
%% ============================================================================

/**
 * validate_no_circular_superiority
 * Detect direct circular superiority relations: sup(A,B) and sup(B,A)
 *
 * IMPORTANT: Only flag cycles between rules with DIFFERENT conclusions.
 * Rules with the same conclusion (e.g., both conclude urgency=emergency)
 * don't actually conflict in DeLP semantics, so cycles between them
 * are harmless and never traversed during reasoning.
 */
validate_no_circular_superiority :-
    forall(
        (   sup(A, B), sup(B, A), A @< B,  % @< prevents duplicates
            % Only flag if rules have contradicting goals (different values)
            rules_have_contradicting_goals(A, B)
        ),
        assertz(validation_error(circular_superiority, [A, B]))
    ).

/**
 * rules_have_contradicting_goals(+RuleId1, +RuleId2)
 * True if two rules conclude different values for the same variable.
 * Rules with the same conclusion don't conflict in DeLP.
 */
rules_have_contradicting_goals(R1, R2) :-
    rule_info(R1, Goal1, _, _),
    rule_info(R2, Goal2, _, _),
    goals_contradict(Goal1, Goal2).

/**
 * validate_no_transitive_cycles
 * Detect longer cycles in superiority: A > B > C > ... > A
 *
 * IMPORTANT: Only flag cycles that contain at least one pair of rules
 * with contradicting conclusions. Cycles between same-value rules
 * are harmless in DeLP semantics.
 */
validate_no_transitive_cycles :-
    findall(Cycle, find_a_cycle(Cycle), Cycles),
    % Remove duplicates and get canonical representation
    maplist(sort, Cycles, CanonicalCycles),
    list_to_set(CanonicalCycles, FinalCycles),
    % Assert one error for each unique cycle that has contradicting rules
    forall(
        (   member(C, FinalCycles),
            C = [_|_],  % Ensure it's a non-empty list
            cycle_has_contradiction(C)  % Only flag if cycle has real conflict
        ),
        assertz(validation_error(transitive_cycle, C))
    ).

/**
 * cycle_has_contradiction(+Cycle)
 * True if the cycle contains at least one pair of consecutive rules
 * that have contradicting goals. A cycle of all same-value rules
 * is harmless and should not be flagged.
 */
cycle_has_contradiction(Cycle) :-
    % Check consecutive pairs in the cycle
    append(_, [R1, R2|_], Cycle),
    rules_have_contradicting_goals(R1, R2),
    !.
cycle_has_contradiction([First|Rest]) :-
    % Also check wrap-around (last -> first)
    last(Rest, Last),
    rules_have_contradicting_goals(Last, First).

% find_a_cycle(-Cycle)
% Finds one cycle in the sup/2 relation graph.
find_a_cycle(Cycle) :-
    % For any edge A -> B
    sup(A, B),
    % Try to find a path from B back to A
    find_sup_path(B, A, [A, B], ReversedPath),
    reverse(ReversedPath, Cycle).

% find_sup_path(+Start, +End, +Visited, -Path)
% Finds a path from Start to End using sup/2 relations,
% avoiding nodes in the Visited list.
find_sup_path(End, End, Path, Path).
find_sup_path(Start, End, Visited, Path) :-
    sup(Start, Next),
    \+ member(Next, Visited),
    find_sup_path(Next, End, [Next|Visited], Path).

/**
 * sup_transitive(+A, +B)
 * Transitive closure of superiority relation
 */
sup_transitive(A, B) :- sup(A, B).
sup_transitive(A, C) :-
    sup(A, B),
    sup_transitive(B, C),
    A \= C.

%% ============================================================================
%% Validation: Contradictory Strict Rules
%% ============================================================================

/**
 * validate_no_contradictory_strict_rules
 * Two strict rules with same variable but different values without superiority
 * This creates an unresolvable conflict
 */
validate_no_contradictory_strict_rules :-
    forall(
        (
            rule_info(R1, Goal1, strict, Body1),
            rule_info(R2, Goal2, strict, Body2),
            R1 @< R2,  % Prevent duplicates
            goals_contradict(Goal1, Goal2),
            bodies_can_both_hold(Body1, Body2),
            \+ sup(R1, R2),
            \+ sup(R2, R1)
        ),
        assertz(validation_error(contradictory_strict_rules, [R1, Goal1, R2, Goal2]))
    ).

/**
 * goals_contradict(+Goal1, +Goal2)
 * Two goals are contradictory if same functor/arity, but different terms
 */
goals_contradict(Goal1, Goal2) :-
    functor(Goal1, Functor, Arity),
    functor(Goal2, Functor, Arity),
    Arity > 0,
    Goal1 \= Goal2.

/**
 * bodies_can_both_hold(+Body1, +Body2)
 * Check if two rule bodies can potentially both be true
 * Conservative: if we can't prove they're exclusive, assume they can both hold
 */
bodies_can_both_hold([], _) :- !.  % Empty body always holds
bodies_can_both_hold(_, []) :- !.  % Empty body always holds
bodies_can_both_hold(Body1, Body2) :-
    % Check if bodies don't contain contradictory literals
    \+ bodies_are_exclusive(Body1, Body2).

/**
 * bodies_are_exclusive(+Body1, +Body2)
 * Two bodies are exclusive if they contain contradictory goals
 */
bodies_are_exclusive(Body1, Body2) :-
    member(Lit1, Body1),
    member(Lit2, Body2),
    literals_contradict(Lit1, Lit2).

/**
 * literals_contradict(+Lit1, +Lit2)
 * Two literals contradict if they're the same predicate with different values
 */
literals_contradict(Lit1, Lit2) :-
    functor(Lit1, Functor, Arity),
    functor(Lit2, Functor, Arity),
    Arity > 0,
    Lit1 \= Lit2.

%% ============================================================================
%% Validation: Superiority References
%% ============================================================================

/**
 * validate_superiority_references
 * Check that all superiority relations reference existing rules
 */
validate_superiority_references :-
    forall(
        (sup(Superior, _), \+ rule_exists(Superior)),
        assertz(validation_error(undefined_superior_rule, Superior))
    ),
    forall(
        (sup(_, Inferior), \+ rule_exists(Inferior)),
        assertz(validation_error(undefined_inferior_rule, Inferior))
    ).

/**
 * rule_exists(+RuleId)
 * Check if rule exists in rule_info
 */
rule_exists(RuleId) :-
    rule_info(RuleId, _, _, _).

%% ============================================================================
%% Validation: Rule Structure
%% ============================================================================

/**
 * validate_rule_structure
 * Check that all rules have valid structure
 */
validate_rule_structure :-
    forall(
        (
            rule_info(RuleId, Goal, Type, Body),
            \+ valid_rule_type(Type)
        ),
        assertz(validation_error(invalid_rule_type, [RuleId, Type]))
    ),
    forall(
        (
            rule_info(RuleId, Goal, _, Body),
            \+ is_list(Body)
        ),
        assertz(validation_error(invalid_rule_body, [RuleId, not_a_list]))
    ),
    forall(
        (
            rule_info(RuleId, Goal, _, _),
            \+ valid_goal_structure(Goal)
        ),
        assertz(validation_error(invalid_goal_structure, [RuleId, Goal]))
    ).

/**
 * valid_rule_type(+Type)
 * Valid rule types
 */
valid_rule_type(strict).
valid_rule_type(defeasible).

/**
 * valid_goal_structure(+Goal)
 * Goal must be a compound term with at least one argument
 */
valid_goal_structure(Goal) :-
    compound(Goal),
    Goal =.. [_Functor|Args],
    Args \= [].

%% ============================================================================
%% Validation: Rule Body References
%% ============================================================================

/**
 * validate_rule_body_references
 * Check that predicates in rule bodies are either:
 * - Another rule's conclusion
 * - A known dynamic predicate (matches, has, like, pattern)
 */
validate_rule_body_references :-
    findall(Var, (rule_info(_, G, _, _), G =.. [Var|_]), AllVars),
    sort(AllVars, KnownVars),
    forall(
        (
            rule_info(RuleId, _, _, Body),
            member(Lit, Body),
            Lit \= true,
            \+ is_negation(Lit),
            Lit =.. [Pred|_],
            \+ member(Pred, KnownVars),
            \+ is_builtin_predicate(Pred)
        ),
        assertz(validation_error(undefined_body_reference, [RuleId, Lit]))
    ).

is_negation(\+(_)).
is_negation(\\+(_)).

is_builtin_predicate(matches).
is_builtin_predicate(has).
is_builtin_predicate(like).
is_builtin_predicate(is_like).
is_builtin_predicate(pattern).
is_builtin_predicate(true).
% Collection quantifiers
is_builtin_predicate(has_any_like).
is_builtin_predicate(has_any_eq).
is_builtin_predicate(has_any_neq).
is_builtin_predicate(has_all_like).
is_builtin_predicate(has_all_eq).
is_builtin_predicate(has_all_neq).
is_builtin_predicate(has_none_like).
is_builtin_predicate(has_none_eq).
is_builtin_predicate(has_none_neq).
is_builtin_predicate(has_any_property_is).
% Length/count predicates
is_builtin_predicate(length_where_like_eq).
is_builtin_predicate(length_where_like_gt).
is_builtin_predicate(length_where_like_lt).
is_builtin_predicate(length_where_like_gte).
is_builtin_predicate(length_where_like_lte).
is_builtin_predicate(length_where_is_eq).
is_builtin_predicate(length_where_is_gt).
is_builtin_predicate(length_where_is_lt).
is_builtin_predicate(length_where_is_gte).
is_builtin_predicate(length_where_is_lte).

%% ============================================================================
%% Validation: Unreachable Rules
%% ============================================================================

/**
 * validate_no_unreachable_rules
 * Detect defeasible rules that can never be warranted
 */
validate_no_unreachable_rules :-
    forall(
        (
            rule_info(RuleId, Goal, defeasible, Body),
            is_always_defeated(RuleId, Goal, Body, Reason)
        ),
        assertz(validation_error(unreachable_defeasible_rule, [RuleId, Reason]))
    ).

/**
 * is_always_defeated(+RuleId, +Goal, +Body, -Reason)
 * Comprehensive check for always-defeated rules
 */
is_always_defeated(_RuleId, Goal, _Body, defeated_by_unconditional_strict(StrictId)) :-
    % Case 1: Defeated by unconditional strict rule
    rule_info(StrictId, StrictGoal, strict, []),
    goals_contradict(Goal, StrictGoal).
    % Strict rules always defeat defeasible rules, no explicit sup needed

is_always_defeated(_RuleId, Goal, Body, defeated_by_more_specific_strict(StrictId)) :-
    % Case 2: Defeated by strict rule with subset of conditions
    rule_info(StrictId, StrictGoal, strict, StrictBody),
    StrictBody \= [],
    goals_contradict(Goal, StrictGoal),
    % Strict rules always defeat defeasible rules
    % Strict body is subset of defeasible body (more specific)
    body_is_subset(StrictBody, Body).

is_always_defeated(RuleId, Goal, _Body, unsatisfiable_body) :-
    % Case 3: Body contains contradiction
    rule_info(RuleId, Goal, _, Body),
    body_is_contradictory(Body).

/**
 * body_is_subset(+Subset, +Superset)
 * Check if all literals in Subset appear in Superset
 */
body_is_subset([], _).
body_is_subset([Lit|Rest], Superset) :-
    member(Lit, Superset),
    body_is_subset(Rest, Superset).

/**
 * body_is_contradictory(+Body)
 * Check if body contains contradictory literals
 */
body_is_contradictory(Body) :-
    member(Lit1, Body),
    member(Lit2, Body),
    Lit1 \= Lit2,
    literals_contradict(Lit1, Lit2).

%% ============================================================================
%% Validation: Semantic Consistency
%% ============================================================================

/**
 * validate_semantic_consistency
 * Check for semantic issues in the program
 */
validate_semantic_consistency :-
    % Check for rules with identical heads and bodies (duplicates)
    validate_no_duplicate_rules,
    % Check for self-referential rules (rule body references its own head)
    validate_no_immediate_self_reference.

/**
 * validate_no_duplicate_rules
 * Detect duplicate rules (same head and body)
 */
validate_no_duplicate_rules :-
    forall(
        (
            rule_info(R1, Goal, Type, Body),
            rule_info(R2, Goal, Type, Body),
            R1 @< R2
        ),
        assertz(validation_error(duplicate_rule, [R1, R2, Goal]))
    ).

/**
 * validate_no_immediate_self_reference
 * Detect rules where body directly references head
 */
validate_no_immediate_self_reference :-
    forall(
        (
            rule_info(RuleId, Goal, _, Body),
            Goal =.. [Functor|_],
            member(Lit, Body),
            Lit =.. [Functor|_]
        ),
        assertz(validation_error(self_referential_rule, [RuleId, Goal]))
    ).

%% ============================================================================
%% Analysis: Complexity Metrics
%% ============================================================================

/**
 * analyze_program_complexity(-Metrics)
 * Compute complexity metrics for the program
 */
analyze_program_complexity(Metrics) :-
    findall(_, rule_info(_, _, _, _), AllRules),
    length(AllRules, TotalRules),
    findall(_, rule_info(_, _, strict, _), StrictRules),
    length(StrictRules, NumStrict),
    findall(_, rule_info(_, _, defeasible, _), DefRules),
    length(DefRules, NumDefeasible),
    findall(_, sup(_, _), SupRels),
    length(SupRels, NumSuperiority),
    findall(Var, (rule_info(_, G, _, _), G =.. [Var|_]), Vars),
    sort(Vars, UniqueVars),
    length(UniqueVars, NumVariables),
    Metrics = _{
        total_rules: TotalRules,
        strict_rules: NumStrict,
        defeasible_rules: NumDefeasible,
        superiority_relations: NumSuperiority,
        variables: NumVariables
    }.

%% ============================================================================
%% Error Reporting
%% ============================================================================

/**
 * print_validation_errors(+Errors)
 * Pretty print validation errors
 */
print_validation_errors([]).
print_validation_errors([error(Type, Details)|Rest]) :-
    format('  [~w] ', [Type]),
    print_error_details(Type, Details),
    nl,
    print_validation_errors(Rest).

/**
 * print_error_details(+Type, +Details)
 * Format-specific error details
 */
print_error_details(circular_superiority, [A, B]) :-
    format('Circular superiority: ~w <-> ~w', [A, B]).

print_error_details(transitive_cycle, Path) :-
    format('Transitive cycle: ~w', [Path]).

print_error_details(contradictory_strict_rules, [R1, G1, R2, G2]) :-
    format('Contradictory strict rules without resolution: ~w (~w) vs ~w (~w)', [R1, G1, R2, G2]).

print_error_details(undefined_superior_rule, RuleId) :-
    format('Superiority references undefined rule: ~w', [RuleId]).

print_error_details(undefined_inferior_rule, RuleId) :-
    format('Superiority references undefined rule: ~w', [RuleId]).

print_error_details(invalid_rule_type, [RuleId, Type]) :-
    format('Rule ~w has invalid type: ~w', [RuleId, Type]).

print_error_details(invalid_rule_body, [RuleId, Reason]) :-
    format('Rule ~w has invalid body: ~w', [RuleId, Reason]).

print_error_details(invalid_goal_structure, [RuleId, Goal]) :-
    format('Rule ~w has invalid goal structure: ~w', [RuleId, Goal]).

print_error_details(undefined_body_reference, [RuleId, Lit]) :-
    format('Rule ~w references undefined predicate in body: ~w', [RuleId, Lit]).

print_error_details(unreachable_defeasible_rule, [RuleId, Reason]) :-
    format('Defeasible rule ~w is unreachable: ~w', [RuleId, Reason]).

print_error_details(duplicate_rule, [R1, R2, Goal]) :-
    format('Duplicate rules ~w and ~w for goal ~w', [R1, R2, Goal]).

print_error_details(self_referential_rule, [RuleId, Goal]) :-
    format('Rule ~w is self-referential: body references ~w', [RuleId, Goal]).

print_error_details(Type, Details) :-
    format('~w: ~w', [Type, Details]).

%% ============================================================================
%% Utility Predicates
%% ============================================================================

/**
 * get_all_rules(-Rules)
 * Get all rule IDs
 */
get_all_rules(Rules) :-
    findall(RuleId, rule_info(RuleId, _, _, _), Rules).

/**
 * get_rules_for_variable(+Variable, -Rules)
 * Get all rules that conclude a specific variable
 */
get_rules_for_variable(Variable, Rules) :-
    findall(
        RuleId,
        (rule_info(RuleId, Goal, _, _), Goal =.. [Variable|_]),
        Rules
    ).

/**
 * get_superiority_graph(-Graph)
 * Get superiority relations as graph structure
 * Graph: list of edge(Superior, Inferior)
 */
get_superiority_graph(Graph) :-
    findall(edge(A, B), sup(A, B), Graph).

/**
 * has_potential_conflicts(+Variable)
 * Check if a variable has multiple rules that could conflict
 */
has_potential_conflicts(Variable) :-
    findall(Value,
            (rule_info(_, Goal, _, _), Goal =.. [Variable, Value]),
            Values),
    sort(Values, UniqueValues),
    length(UniqueValues, N),
    N > 1.

%% ============================================================================
%% DSL Analysis Support - Issue Formatting
%% ============================================================================
%%
%% NOTE: Conflict detection has been moved to delp_meta.pl which uses
%% mutual exclusivity analysis for accurate conflict detection.
%% This module now only provides issue formatting for validation errors.

/**
 * format_issue_safe(-Issue)
 * Format validation errors as issues with safe serialization.
 * Called from Python via: findall(Issue, format_issue_safe(Issue), Issues)
 */
format_issue_safe(Issue) :-
    validation_error(Type, Details),
    format_issue_message(Type, Details, Message),
    term_to_atom(Details, DetailsAtom),
    Issue = _{type: Type, message: Message, details: DetailsAtom}.

format_issue_message(unreachable_defeasible_rule, [RuleId, Reason], Message) :-
    !,
    format(atom(Message), 'Rule ~w can never win: ~w', [RuleId, Reason]).

format_issue_message(circular_superiority, [A, B], Message) :-
    !,
    format(atom(Message), 'Circular override between ~w and ~w', [A, B]).

format_issue_message(contradictory_strict_rules, [R1, _, R2, _], Message) :-
    !,
    format(atom(Message), 'Strict rules ~w and ~w contradict without resolution', [R1, R2]).

format_issue_message(Type, Details, Message) :-
    format(atom(Message), '~w: ~w', [Type, Details]).
