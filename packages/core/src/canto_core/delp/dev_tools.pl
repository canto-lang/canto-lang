/**
 * Developer Tools for DeLP
 *
 * Build-time analysis tools for debugging and validating DeLP programs.
 * Separate from the core meta-interpreter to keep concerns isolated.
 *
 * Features:
 * - Assumption Visibility: Track and validate symbolic assumptions
 * - (Future) Trace Mode: Show reasoning steps
 * - (Future) What-If Testing: Test with sample inputs
 *
 * Usage:
 *   ?- consult('dev_tools.pl').
 *   ?- clear_assumption_warnings.
 *   ... run queries ...
 *   ?- get_assumption_warnings(W).
 */

:- dynamic assumption_warning/3.  % assumption_warning(Type, Predicate, Suggestion)
:- dynamic trace_enabled/0.       % Flag to enable/disable tracing
:- dynamic trace_event/3.         % trace_event(Step, Type, Details)

%% ============================================================================
%% Assumption Visibility
%% ============================================================================
%%
%% At build time, predicates like matches(X, Category) are symbolic premises.
%% This module validates that referenced symbols (categories, variables) exist.
%%
%% Warnings are generated for:
%% - matches(X, unknown_category) - category not defined in pattern/2
%% - unknown_predicate(X, Y) - predicate functor not recognized

/**
 * clear_assumption_warnings/0
 * Clear all recorded assumption warnings (call before new analysis)
 */
clear_assumption_warnings :-
    retractall(assumption_warning(_, _, _)).

/**
 * validate_assumption(+Literal)
 * Validate an assumption and record warnings if problems found.
 */
validate_assumption(Literal) :-
    Literal =.. [Functor|Args],
    validate_by_type(Functor, Args, Literal).

/**
 * validate_by_type(+Functor, +Args, +Literal)
 * Type-specific validation
 */
% matches(X, Category) - check if Category is defined in pattern/2
validate_by_type(matches, [_, Category], Literal) :-
    !,
    (   atom(Category),
        \+ pattern(Category, _),
        \+ assumption_warning(unknown_category, Literal, _)
    ->  find_similar_category(Category, Suggestion),
        assertz(assumption_warning(unknown_category, Literal, Suggestion))
    ;   true
    ).

% has/like are expected symbolic premises
validate_by_type(has, _, _) :- !.
validate_by_type(like, _, _) :- !.

% Known variable predicates (from rule_info) are expected
validate_by_type(Functor, [_], _) :-
    rule_info(_, Goal, _, _),
    Goal =.. [Functor|_],
    !.

% Unknown predicate - warn (always succeeds to not block argument construction)
validate_by_type(Functor, Args, Literal) :-
    (   assumption_warning(unknown_predicate, Literal, _)
    ->  true  % Already warned, don't duplicate
    ;   length(Args, Arity),
        find_similar_predicate(Functor, Arity, Suggestion),
        assertz(assumption_warning(unknown_predicate, Literal, Suggestion))
    ).

/**
 * find_similar_category(+Category, -Suggestion)
 * Find similar category name for "did you mean" suggestions
 */
find_similar_category(Category, Suggestion) :-
    findall(Cat, pattern(Cat, _), AllCategories),
    sort(AllCategories, Categories),
    find_closest(Category, Categories, Suggestion),
    !.
find_similar_category(_, none).

/**
 * find_similar_predicate(+Functor, +Arity, -Suggestion)
 */
find_similar_predicate(Functor, _, Suggestion) :-
    findall(F, (rule_info(_, Goal, _, _), Goal =.. [F|_]), RuleFunctors),
    append([matches, has, like, pattern], RuleFunctors, AllFunctors),
    sort(AllFunctors, Functors),
    find_closest(Functor, Functors, Suggestion),
    !.
find_similar_predicate(_, _, none).

/**
 * find_closest(+Target, +Candidates, -Match)
 * Simple similarity matching for suggestions
 */
find_closest(Target, Candidates, Match) :-
    atom_string(Target, TargetStr),
    string_length(TargetStr, TargetLen),
    find_best(TargetStr, TargetLen, Candidates, 3, none, Match),
    Match \= none.

find_best(_, _, [], _, Best, Best).
find_best(TargetStr, TargetLen, [Cand|Rest], BestDist, BestMatch, Final) :-
    atom_string(Cand, CandStr),
    string_length(CandStr, CandLen),
    % Quick length check - skip if too different
    LenDiff is abs(TargetLen - CandLen),
    (   LenDiff > 2
    ->  find_best(TargetStr, TargetLen, Rest, BestDist, BestMatch, Final)
    ;   simple_distance(TargetStr, CandStr, Dist),
        (   Dist < BestDist, Dist =< 2
        ->  find_best(TargetStr, TargetLen, Rest, Dist, Cand, Final)
        ;   find_best(TargetStr, TargetLen, Rest, BestDist, BestMatch, Final)
        )
    ).

/**
 * simple_distance(+S1, +S2, -Dist)
 * Simplified string distance (character differences)
 */
simple_distance(S1, S2, Dist) :-
    string_chars(S1, C1),
    string_chars(S2, C2),
    count_differences(C1, C2, 0, Dist).

count_differences([], [], Acc, Acc).
count_differences([], Rest, Acc, Dist) :-
    length(Rest, L),
    Dist is Acc + L.
count_differences(Rest, [], Acc, Dist) :-
    length(Rest, L),
    Dist is Acc + L.
count_differences([H|T1], [H|T2], Acc, Dist) :-
    !,
    count_differences(T1, T2, Acc, Dist).
count_differences([_|T1], [_|T2], Acc, Dist) :-
    Acc1 is Acc + 1,
    count_differences(T1, T2, Acc1, Dist).

/**
 * get_assumption_warnings(-Warnings)
 * Get all warnings as list of dicts (Janus-friendly)
 */
get_assumption_warnings(Warnings) :-
    findall(
        _{type: Type, predicate: PredAtom, suggestion: Sugg},
        (   assumption_warning(Type, Pred, Sugg),
            term_to_atom(Pred, PredAtom)
        ),
        Warnings
    ).

/**
 * print_assumption_warnings/0
 * Pretty-print all warnings
 */
print_assumption_warnings :-
    (   assumption_warning(_, _, _)
    ->  format('~n~`=t ASSUMPTION WARNINGS ~`=t~60|~n', []),
        forall(
            assumption_warning(Type, Pred, Suggestion),
            print_warning(Type, Pred, Suggestion)
        )
    ;   format('~n✓ No assumption warnings~n', [])
    ).

print_warning(unknown_category, matches(_, Category), Suggestion) :-
    format('  ✗ Unknown category: ~w~n', [Category]),
    (   Suggestion \= none
    ->  format('    → Did you mean: ~w?~n', [Suggestion])
    ;   true
    ).

print_warning(unknown_predicate, Pred, Suggestion) :-
    Pred =.. [Functor|_],
    format('  ✗ Unknown predicate: ~w~n', [Functor]),
    (   Suggestion \= none
    ->  format('    → Did you mean: ~w?~n', [Suggestion])
    ;   true
    ).

%% ============================================================================
%% Trace Mode for Dialectical Trees
%% ============================================================================
%%
%% Provides step-by-step visibility into the reasoning process:
%% - Which arguments were found
%% - How defeaters were identified
%% - The marking algorithm progression
%% - Why arguments were marked defeated/undefeated/blocked
%%
%% Usage:
%%   ?- enable_trace.
%%   ?- delp_query(goal, Status, Tree).
%%   ?- get_trace(Events).
%%   ?- print_trace.
%%   ?- disable_trace.

/**
 * enable_trace/0
 * Enable trace mode - events will be recorded during reasoning
 */
enable_trace :-
    retractall(trace_enabled),
    assertz(trace_enabled),
    clear_trace.

/**
 * disable_trace/0
 * Disable trace mode
 */
disable_trace :-
    retractall(trace_enabled).

/**
 * clear_trace/0
 * Clear all recorded trace events
 */
clear_trace :-
    retractall(trace_event(_, _, _)),
    nb_setval(trace_step, 0).

/**
 * trace_step(-Step)
 * Get and increment the current trace step counter
 */
trace_step(Step) :-
    (   nb_current(trace_step, Current)
    ->  Step is Current + 1,
        nb_setval(trace_step, Step)
    ;   Step = 1,
        nb_setval(trace_step, 1)
    ).

/**
 * trace/2
 * Record a trace event if tracing is enabled
 * trace(+Type, +Details)
 */
trace(Type, Details) :-
    (   trace_enabled
    ->  trace_step(Step),
        assertz(trace_event(Step, Type, Details))
    ;   true
    ).

/**
 * Trace event types and their recording predicates
 */

% Called when starting to find arguments for a goal
trace_query_start(Goal) :-
    term_to_atom(Goal, GoalAtom),
    trace(query_start, _{goal: GoalAtom}).

% Called when an argument is found
trace_argument_found(Goal, RuleId, _Premises, Specificity) :-
    term_to_atom(Goal, GoalAtom),
    trace(argument_found, _{
        goal: GoalAtom,
        rule_id: RuleId,
        specificity: Specificity
    }).

% Called when no arguments are found for a goal
trace_no_arguments(Goal) :-
    term_to_atom(Goal, GoalAtom),
    trace(no_arguments, _{goal: GoalAtom}).

% Called when starting to build dialectical tree
trace_tree_build_start(Argument) :-
    Argument = argument(Goal, RuleId, _, Spec),
    term_to_atom(Goal, GoalAtom),
    trace(tree_build_start, _{
        goal: GoalAtom,
        rule_id: RuleId,
        specificity: Spec
    }).

% Called when a defeater is found
trace_defeater_found(Argument, Defeater, DefeatType) :-
    Argument = argument(Goal1, RuleId1, _, Spec1),
    Defeater = argument(Goal2, RuleId2, _, Spec2),
    term_to_atom(Goal1, Goal1Atom),
    term_to_atom(Goal2, Goal2Atom),
    % Determine defeat reason with details
    defeat_reason_details(RuleId1, RuleId2, Spec1, Spec2, ReasonType, ReasonDetails),
    trace(defeater_found, _{
        argument_goal: Goal1Atom,
        argument_rule: RuleId1,
        argument_specificity: Spec1,
        defeater_goal: Goal2Atom,
        defeater_rule: RuleId2,
        defeater_specificity: Spec2,
        defeat_type: DefeatType,
        reason_type: ReasonType,
        reason_details: ReasonDetails
    }).

/**
 * defeat_reason_details(+RuleId1, +RuleId2, +Spec1, +Spec2, -ReasonType, -Details)
 * Determine why the defeater defeats the argument with detailed explanation
 */
defeat_reason_details(RuleId1, RuleId2, _Spec1, _Spec2, explicit_superiority, Details) :-
    sup(RuleId2, RuleId1),
    !,
    format(atom(Details), 'sup(~w, ~w) declared', [RuleId2, RuleId1]).

defeat_reason_details(_RuleId1, _RuleId2, Spec1, Spec2, more_specific, Details) :-
    Spec2 > Spec1,
    !,
    format(atom(Details), 'defeater score ~w > argument score ~w', [Spec2, Spec1]).

defeat_reason_details(_RuleId1, _RuleId2, Spec1, Spec2, blocking, Details) :-
    format(atom(Details), 'equal specificity (~w vs ~w), no superiority declared', [Spec1, Spec2]).

% Called when a node is marked with a status
trace_marking(Argument, Status, Reason) :-
    Argument = argument(Goal, RuleId, _, _),
    term_to_atom(Goal, GoalAtom),
    term_to_atom(Reason, ReasonAtom),
    trace(marking, _{
        goal: GoalAtom,
        rule_id: RuleId,
        status: Status,
        reason: ReasonAtom
    }).

% Called when final status is determined
trace_final_status(Goal, Status, WinningRule) :-
    term_to_atom(Goal, GoalAtom),
    trace(final_status, _{
        goal: GoalAtom,
        status: Status,
        winning_rule: WinningRule
    }).

/**
 * get_trace(-Events)
 * Get all trace events as a list of dicts (Janus-friendly)
 */
get_trace(Events) :-
    findall(
        _{step: Step, type: Type, details: Details},
        trace_event(Step, Type, Details),
        Events
    ).

/**
 * print_trace/0
 * Pretty-print the trace
 */
print_trace :-
    (   trace_event(_, _, _)
    ->  format('~n~`=t REASONING TRACE ~`=t~60|~n', []),
        forall(
            trace_event(Step, Type, Details),
            print_trace_event(Step, Type, Details)
        ),
        format('~`=t~60|~n', [])
    ;   format('~n(No trace events recorded. Did you enable_trace?)~n', [])
    ).

print_trace_event(Step, query_start, Details) :-
    Goal = Details.goal,
    format('[~w] QUERY: ~w~n', [Step, Goal]).

print_trace_event(Step, argument_found, Details) :-
    format('[~w] ARGUMENT FOUND:~n', [Step]),
    format('      Goal: ~w~n', [Details.goal]),
    format('      Rule: ~w~n', [Details.rule_id]),
    format('      Specificity: ~w~n', [Details.specificity]).

print_trace_event(Step, no_arguments, Details) :-
    format('[~w] NO ARGUMENTS for ~w~n', [Step, Details.goal]).

print_trace_event(Step, tree_build_start, Details) :-
    format('[~w] BUILDING TREE for ~w via ~w~n',
           [Step, Details.goal, Details.rule_id]).

print_trace_event(Step, defeater_found, Details) :-
    format('[~w] DEFEATER FOUND:~n', [Step]),
    format('      ~w (~w, spec=~w)~n', [Details.argument_goal, Details.argument_rule, Details.argument_specificity]),
    format('      defeated by: ~w (~w, spec=~w)~n', [Details.defeater_goal, Details.defeater_rule, Details.defeater_specificity]),
    format('      type: ~w~n', [Details.defeat_type]),
    format('      reason: ~w - ~w~n', [Details.reason_type, Details.reason_details]).

print_trace_event(Step, marking, Details) :-
    format('[~w] MARKING: ~w (~w) = ~w~n',
           [Step, Details.goal, Details.rule_id, Details.status]),
    (   Details.reason \= none
    ->  format('      reason: ~w~n', [Details.reason])
    ;   true
    ).

print_trace_event(Step, final_status, Details) :-
    format('[~w] FINAL STATUS: ~w = ~w~n', [Step, Details.goal, Details.status]),
    (   Details.winning_rule \= none
    ->  format('      winning rule: ~w~n', [Details.winning_rule])
    ;   true
    ).

print_trace_event(Step, Type, Details) :-
    format('[~w] ~w: ~w~n', [Step, Type, Details]).
