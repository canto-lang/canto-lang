/**
 * DeLP Meta-Interpreter
 * Implements proper Defeasible Logic Programming semantics
 * Based on García & Simari's DeLP framework
 *
 * Key concepts:
 * - Arguments: Sets of rules that derive a conclusion
 * - Defeat: Conflict resolution via specificity and explicit priority
 * - Blocking: Mutual defeat when no preference exists
 * - Dialectical trees: Argumentation structure for warrant computation
 *
 * Specificity Calculation (Information-Theoretic):
 * - Arguments are compared by semantic specificity, not just body length
 * - matches(X, Category) → score = 1000 / num_patterns (smaller = more specific)
 * - Explicit category_weight/2 facts can override automatic calculation
 * - This ensures a rule matching "specific_vaccines" (5 patterns) beats
 *   a rule with two generic "has" conditions
 *
 * Usage via Janus:
 *   delp_query(vaccine_flag(true), Status, Tree).
 *   Status: warranted | defeated | undecided | blocked
 */

:- dynamic gensym_counter/2.
:- dynamic rule_info/4.
:- dynamic sup/2.
:- dynamic pattern/2.
:- dynamic category_weight/2.  % Optional explicit weights for categories
:- dynamic assumption_warning/3.  % assumption_warning(Type, Predicate, Suggestion)
:- discontiguous build_dialectical_tree/4.  % Clauses split by helper predicates

% Load mutual exclusivity detection (conflict analysis)
:- consult('delp_mutual_exclusivity.pl').

%% ============================================================================
%% 0. Symbolic Predicates for Build-Time Analysis
%% ============================================================================
%%
%% These predicates provide symbolic semantics for DSL conditions.
%% At build time, we don't evaluate actual text content - we analyze
%% rule structure, conflicts, and specificity.
%%
%% matches(Var, Category) succeeds if Category is defined with patterns.
%% has(Var, Property) succeeds unconditionally (property attribution).
%% like(Var, Pattern) succeeds unconditionally (pattern similarity).
%%
%% This allows arguments to be constructed with these as facts rather
%% than assumptions, enabling proper dialectical tree analysis.

/**
 * matches(+Var, +Category)
 * Symbolic semantic matching - succeeds if Category has defined patterns.
 *
 * At build time, this doesn't check actual text content.
 * It verifies that the semantic category exists and is well-defined.
 * Specificity is computed based on pattern count (fewer = more specific).
 */
matches(_, Category) :-
    ground(Category),
    pattern(Category, _),
    !.

/**
 * has(+Var, +Property)
 * Symbolic property attribution - always succeeds at build time.
 *
 * Represents that a variable has a certain property.
 * The actual property check happens at LLM runtime.
 */
has(_, _).

/**
 * like(+Var, +Pattern)
 * Symbolic pattern similarity - always succeeds at build time.
 *
 * Represents fuzzy/semantic similarity matching.
 * The actual similarity check happens at LLM runtime.
 */
like(_, _).

/**
 * is_like(+Path, +Category)
 * Symbolic path-based semantic matching - succeeds if Category has patterns.
 *
 * Similar to matches/2 but used for nested path expressions.
 * Example: is_like(text_of_('?symptoms', '?patient'), emergency_symptoms)
 */
is_like(_, Category) :-
    ground(Category),
    pattern(Category, _),
    !.
is_like(_, _).  % Fallback for unground categories

%% ----------------------------------------------------------------------------
%% Collection Quantifier Predicates (has_any_*, has_all_*, has_none_*)
%% ----------------------------------------------------------------------------
%% These predicates represent quantified conditions over collections.
%% At build time, they succeed symbolically to enable argument construction.
%% At runtime, the LLM evaluates actual collection membership.

/**
 * has_any_like(+Collection, +Category)
 * True if ANY item in collection matches the semantic category.
 */
has_any_like(_, Category) :-
    ground(Category),
    pattern(Category, _),
    !.
has_any_like(_, _).

/**
 * has_any_eq(+Collection, +Value)
 * True if ANY item in collection equals the value.
 */
has_any_eq(_, _).

/**
 * has_any_neq(+Collection, +Value)
 * True if ANY item in collection does not equal the value.
 */
has_any_neq(_, _).

/**
 * has_all_like(+Collection, +Category)
 * True if ALL items in collection match the semantic category.
 */
has_all_like(_, Category) :-
    ground(Category),
    pattern(Category, _),
    !.
has_all_like(_, _).

/**
 * has_all_eq(+Collection, +Value)
 * True if ALL items in collection equal the value.
 */
has_all_eq(_, _).

/**
 * has_all_neq(+Collection, +Value)
 * True if ALL items in collection do not equal the value.
 */
has_all_neq(_, _).

/**
 * has_none_like(+Collection, +Category)
 * True if NO items in collection match the semantic category.
 */
has_none_like(_, Category) :-
    ground(Category),
    pattern(Category, _),
    !.
has_none_like(_, _).

/**
 * has_none_eq(+Collection, +Value)
 * True if NO items in collection equal the value.
 */
has_none_eq(_, _).

/**
 * has_none_neq(+Collection, +Value)
 * True if NO items in collection do not equal the value.
 */
has_none_neq(_, _).

/**
 * has_any_property_is(+Collection, +PropertySpec)
 * True if ANY item in collection has a property matching the spec.
 * PropertySpec is typically a dict like {property: X, value: Y}
 */
has_any_property_is(_, _).

%% ----------------------------------------------------------------------------
%% Length/Count Predicates (length_where_*)
%% ----------------------------------------------------------------------------
%% These predicates represent cardinality constraints on filtered collections.
%% At build time, they succeed symbolically.

/**
 * length_where_like_eq(+Collection, +Category, +Count, +Comparison)
 * True if count of items matching Category equals Count.
 */
length_where_like_eq(_, Category, _, _) :-
    ground(Category),
    pattern(Category, _),
    !.
length_where_like_eq(_, _, _, _).

/**
 * length_where_like_gt(+Collection, +Category, +Count, +Comparison)
 * True if count of items matching Category is greater than Count.
 */
length_where_like_gt(_, Category, _, _) :-
    ground(Category),
    pattern(Category, _),
    !.
length_where_like_gt(_, _, _, _).

/**
 * length_where_like_lt(+Collection, +Category, +Count, +Comparison)
 * True if count of items matching Category is less than Count.
 */
length_where_like_lt(_, Category, _, _) :-
    ground(Category),
    pattern(Category, _),
    !.
length_where_like_lt(_, _, _, _).

/**
 * length_where_like_gte(+Collection, +Category, +Count, +Comparison)
 * True if count of items matching Category is >= Count.
 */
length_where_like_gte(_, Category, _, _) :-
    ground(Category),
    pattern(Category, _),
    !.
length_where_like_gte(_, _, _, _).

/**
 * length_where_like_lte(+Collection, +Category, +Count, +Comparison)
 * True if count of items matching Category is <= Count.
 */
length_where_like_lte(_, Category, _, _) :-
    ground(Category),
    pattern(Category, _),
    !.
length_where_like_lte(_, _, _, _).

/**
 * length_where_is_eq(+Collection, +Property, +Value, +Count)
 * True if count of items where Property equals Value equals Count.
 */
length_where_is_eq(_, _, _, _).

/**
 * length_where_is_gt(+Collection, +Property, +Value, +Count)
 * True if count of items where Property equals Value is > Count.
 */
length_where_is_gt(_, _, _, _).

/**
 * length_where_is_lt(+Collection, +Property, +Value, +Count)
 * True if count of items where Property equals Value is < Count.
 */
length_where_is_lt(_, _, _, _).

/**
 * length_where_is_gte(+Collection, +Property, +Value, +Count)
 * True if count of items where Property equals Value is >= Count.
 */
length_where_is_gte(_, _, _, _).

/**
 * length_where_is_lte(+Collection, +Property, +Value, +Count)
 * True if count of items where Property equals Value is <= Count.
 */
length_where_is_lte(_, _, _, _).

%% ============================================================================
%% 1. Main Query Interface
%% ============================================================================

/**
 * delp_query(+Goal, -Status, -Tree)
 * Main entry point for DeLP queries
 *
 * Status values:
 *   - warranted: Goal has an undefeated argument
 *   - defeated: All arguments for Goal are defeated
 *   - undecided: Some arguments have undecided status
 *   - blocked: Arguments exist but are blocked by equally strong counter-arguments
 */
delp_query(Goal, Status, Tree) :-
    % Trace: query starting
    trace_query_start(Goal),
    % Find all arguments for the goal
    findall(Arg, find_argument(Goal, Arg, []), Arguments),
    (   Arguments = [] ->
        trace_no_arguments(Goal),
        Status = undecided,
        Tree = no_arguments(Goal)
    ;   % Trace: arguments found
        trace_arguments_found(Arguments),
        % Build dialectical trees for each argument
        build_all_trees(Arguments, Trees),
        % Mark trees and determine status
        mark_trees(Trees, MarkedTrees),
        determine_status(MarkedTrees, Status, Tree),
        % Trace: final status
        extract_winning_rule(Tree, WinningRule),
        trace_final_status(Goal, Status, WinningRule)
    ).

% Helper to trace all found arguments
trace_arguments_found([]).
trace_arguments_found([Arg|Rest]) :-
    Arg = argument(Goal, RuleId, Premises, Specificity),
    trace_argument_found(Goal, RuleId, Premises, Specificity),
    trace_arguments_found(Rest).

% Extract winning rule from tree
extract_winning_rule(tree(argument(_, RuleId, _, _), _, _), RuleId) :- !.
extract_winning_rule(_, none).

/**
 * determine_status(+MarkedTrees, -Status, -Tree)
 * Determine overall status from marked trees
 */
determine_status(MarkedTrees, Status, Tree) :-
    (   member(tree(Arg, undefeated, DefTree), MarkedTrees) ->
        Status = warranted,
        Tree = tree(Arg, undefeated, DefTree)
    ;   member(tree(_, blocked, _), MarkedTrees),
        \+ member(tree(_, undefeated, _), MarkedTrees) ->
        Status = blocked,
        MarkedTrees = [Tree|_]
    ;   member(tree(_, undecided, _), MarkedTrees) ->
        Status = undecided,
        MarkedTrees = [Tree|_]
    ;   Status = defeated,
        MarkedTrees = [Tree|_]
    ).

/**
 * warranted(+Goal)
 * True if goal is warranted (has an undefeated argument)
 */
warranted(Goal) :-
    delp_query(Goal, warranted, _).

/**
 * defeated(+Goal)
 * True if goal is defeated (all arguments are defeated)
 */
defeated(Goal) :-
    delp_query(Goal, Status, _),
    member(Status, [defeated, blocked]).

%% ============================================================================
%% 2. Argument Construction
%% ============================================================================

/**
 * find_argument(+Goal, -Argument, +Visited)
 * Find an argument for a goal
 * Argument structure: argument(Goal, RuleId, Premises, SpecificityScore)
 *
 * Visited: List of goals already visited (cycle prevention)
 * SpecificityScore: Computed based on semantic information content, not just body length
 */
find_argument(Goal, argument(Goal, RuleId, Premises, SpecificityScore), Visited) :-
    % Cycle check using parameter, not global state
    \+ member(Goal, Visited),

    % Find a rule from metadata that concludes Goal
    rule_info(RuleId, Goal, _Type, BodyList),

    % Calculate specificity score based on semantic content
    calculate_specificity(BodyList, SpecificityScore),

    % Build premises from body list
    build_premises(BodyList, Premises, [Goal|Visited]),

    % Verify argument consistency
    argument_is_consistent(argument(Goal, RuleId, Premises, SpecificityScore)).

/**
 * build_premises(+BodyList, -Premises, +Visited)
 * Build argument premises from rule body
 */
build_premises([], [], _).
build_premises([true|Rest], Premises, Visited) :-
    !,
    build_premises(Rest, Premises, Visited).
build_premises([Literal|Rest], [Premise|Premises], Visited) :-
    build_single_premise(Literal, Premise, Visited),
    build_premises(Rest, Premises, Visited).

/**
 * build_single_premise(+Literal, -Premise, +Visited)
 * Build a single premise
 */
build_single_premise(Literal, Premise, Visited) :-
    % Handle negation as failure (NAF)
    (   Literal = \+(Goal) ->
        (   % Check if negated goal can be derived
            find_argument(Goal, _, Visited) ->
            % Negation fails if goal is derivable
            fail
        ;   Premise = negation(Goal)
        )
    ;   Literal = \\+(Goal) ->
        (   find_argument(Goal, _, Visited) ->
            fail
        ;   Premise = negation(Goal)
        )
    % Handle warrant-based negation (NOT WARRANTED)
    ;   Literal = not_warranted(Goal) ->
        (   % Check if goal is warranted (has undefeated argument)
            check_not_warranted(Goal, Visited) ->
            Premise = not_warranted_premise(Goal)
        ;   fail  % Goal is warranted, so not_warranted fails
        )
    ;   % Try to build argument for this literal
        (   find_argument(Literal, Arg, Visited) ->
            Premise = Arg
        ;   % Check if it's a ground fact or assumption
            (   is_ground_predicate(Literal) ->
                % Use catch to handle undefined predicates gracefully
                (   catch(call(Literal), _, fail) ->
                    Premise = fact(Literal)
                ;   Premise = assumption(Literal),
                    validate_assumption(Literal)
                )
            ;   % Otherwise treat as assumption
                Premise = assumption(Literal),
                validate_assumption(Literal)
            )
        )
    ).

/**
 * check_not_warranted(+Goal, +Visited)
 * Succeeds if Goal is NOT warranted (no undefeated argument exists)
 *
 * This implements warrant-based negation:
 * - If Goal has no arguments → succeeds (not warranted)
 * - If Goal has arguments but all are defeated → succeeds (not warranted)
 * - If Goal has an undefeated argument → fails (is warranted)
 *
 * Cycle detection: If Goal is in Visited, we're in a cycle.
 * Cycles are treated as "blocked" - conservatively fail.
 */
check_not_warranted(Goal, Visited) :-
    % Cycle detection: if we're already evaluating this goal, fail conservatively
    (   member(Goal, Visited) ->
        fail  % Cycle detected - treat as blocked
    ;   % No cycle - perform warrant check
        \+ goal_is_warranted(Goal)
    ).

/**
 * goal_is_warranted(+Goal)
 * Succeeds if Goal has at least one undefeated argument
 */
goal_is_warranted(Goal) :-
    delp_query(Goal, Status, _),
    Status = warranted.

/**
 * is_ground_predicate(+Literal)
 * Check if literal is a known ground predicate that can be called safely.
 * This includes all symbolic predicates defined for build-time analysis.
 */
is_ground_predicate(Literal) :-
    ground(Literal),
    Literal =.. [Pred|_],
    symbolic_predicate(Pred).

/**
 * symbolic_predicate(+Name)
 * Registry of predicates that succeed symbolically at build time.
 */
symbolic_predicate(matches).
symbolic_predicate(has).
symbolic_predicate(like).
symbolic_predicate(pattern).
symbolic_predicate(is_like).
% Collection quantifiers
symbolic_predicate(has_any_like).
symbolic_predicate(has_any_eq).
symbolic_predicate(has_any_neq).
symbolic_predicate(has_all_like).
symbolic_predicate(has_all_eq).
symbolic_predicate(has_all_neq).
symbolic_predicate(has_none_like).
symbolic_predicate(has_none_eq).
symbolic_predicate(has_none_neq).
symbolic_predicate(has_any_property_is).
% Length/count predicates
symbolic_predicate(length_where_like_eq).
symbolic_predicate(length_where_like_gt).
symbolic_predicate(length_where_like_lt).
symbolic_predicate(length_where_like_gte).
symbolic_predicate(length_where_like_lte).
symbolic_predicate(length_where_is_eq).
symbolic_predicate(length_where_is_gt).
symbolic_predicate(length_where_is_lt).
symbolic_predicate(length_where_is_gte).
symbolic_predicate(length_where_is_lte).

/**
 * argument_is_consistent(+Argument)
 * Check that an argument's premises are internally consistent
 * (no contradictory conclusions in sub-arguments)
 */
argument_is_consistent(argument(_, _, Premises, _)) :-
    extract_conclusions(Premises, Conclusions),
    \+ has_contradiction(Conclusions).

/**
 * extract_conclusions(+Premises, -Conclusions)
 * Extract all conclusions from nested arguments
 */
extract_conclusions([], []).
extract_conclusions([Premise|Rest], Conclusions) :-
    (   Premise = argument(Goal, _, SubPremises, _) ->
        extract_conclusions(SubPremises, SubConclusions),
        Conclusions = [Goal|RestConclusions],
        extract_conclusions(Rest, RestConclusions0),
        append(SubConclusions, RestConclusions0, RestConclusions)
    ;   extract_conclusions(Rest, Conclusions)
    ).

/**
 * has_contradiction(+Conclusions)
 * Check if list of conclusions contains contradictory goals
 */
has_contradiction(Conclusions) :-
    member(G1, Conclusions),
    member(G2, Conclusions),
    G1 \= G2,
    contradicts(G1, G2).

%% ============================================================================
%% 3. Defeat Relations (Proper DeLP Semantics)
%% ============================================================================

/**
 * find_defeaters(+Argument, -Defeaters)
 * Find all arguments that defeat the given argument
 *
 * Types of defeat:
 * 1. Proper defeat: Defeater is strictly preferred
 * 2. Blocking defeat: Neither is preferred (mutual defeat)
 */
find_defeaters(Argument, Defeaters) :-
    Argument = argument(_Goal, _, _, _),
    findall(Defeater,
            find_single_defeater(Argument, Defeater),
            AllDefeaters),
    % Remove duplicates
    sort(AllDefeaters, Defeaters).

/**
 * find_single_defeater(+Argument, -Defeater)
 * Find a single defeating argument
 */
find_single_defeater(Argument, Defeater) :-
    Argument = argument(Goal, RuleId, _, _),
    % Find a rule that concludes a contradictory goal
    rule_info(DefRuleId, CounterGoal, _Type, _Body),
    DefRuleId \= RuleId,
    contradicts(Goal, CounterGoal),

    % Build the counter-argument
    find_argument(CounterGoal, Defeater, []),
    Defeater = argument(_, DefRuleId2, _, _),

    % Check defeat relation
    defeats_or_blocks(Defeater, Argument, DefRuleId2, RuleId).

/**
 * defeats_or_blocks(+Arg1, +Arg2, +RuleId1, +RuleId2)
 * True if Arg1 defeats or blocks Arg2
 *
 * Defeat hierarchy:
 * 1. Explicit superiority (sup/2) - highest priority
 * 2. Specificity (more conditions = more specific)
 * 3. Blocking (neither preferred, both have equal specificity)
 *
 * Note: If Arg2 is more specific than Arg1, Arg1 is NOT a defeater
 */
defeats_or_blocks(Arg1, Arg2, RuleId1, RuleId2) :-
    % Check explicit superiority first
    (   sup(RuleId1, RuleId2) ->
        true  % Proper defeat via explicit priority
    ;   sup(RuleId2, RuleId1) ->
        fail  % Arg2 is superior, so Arg1 doesn't defeat
    ;   % No explicit priority - check specificity both ways
        (   is_more_specific(Arg1, Arg2) ->
            true  % Proper defeat via specificity
        ;   is_more_specific(Arg2, Arg1) ->
            fail  % Arg2 is more specific, so Arg1 is NOT a defeater
        ;   % Neither is preferred - genuine blocking defeat
            true
        )
    ).

/**
 * is_more_specific(+Arg1, +Arg2)
 * Arg1 is more specific than Arg2 if it has a higher specificity score
 *
 * Specificity is based on semantic information content:
 * - Matching narrow semantic categories (fewer patterns) = more specific
 * - Explicit weights can override automatic calculation
 * - Body length is only used as a tiebreaker
 */
is_more_specific(argument(_, _, _, Score1), argument(_, _, _, Score2)) :-
    Score1 > Score2.

%% ============================================================================
%% Specificity Calculation (Information-Theoretic)
%% ============================================================================

/**
 * calculate_specificity(+BodyList, -Score)
 * Calculate specificity score for a rule body based on semantic content
 *
 * Scoring algorithm:
 * - matches(X, Category) → inverse of category size (smaller = more specific)
 * - has/like predicates → base score
 * - Variable value checks → base score
 * - Negations → small contribution
 *
 * This ensures that a rule matching a narrow semantic category
 * (e.g., 5 vaccine terms) is more specific than a rule with
 * multiple generic conditions.
 */
calculate_specificity([], 0).
calculate_specificity([Lit|Rest], TotalScore) :-
    literal_specificity(Lit, LitScore),
    calculate_specificity(Rest, RestScore),
    TotalScore is LitScore + RestScore.

/**
 * literal_specificity(+Literal, -Score)
 * Calculate specificity contribution of a single literal
 */
% Skip 'true' literals
literal_specificity(true, 0) :- !.

% matches(Var, Category) - specificity based on category size
% Smaller categories are more specific
literal_specificity(matches(_, Category), Score) :-
    !,
    category_specificity(Category, Score).

% has(Var, Property) - moderate base specificity
literal_specificity(has(_, _), 10) :- !.

% like(Var, Pattern) - moderate base specificity
literal_specificity(like(_, _), 10) :- !.

% Negation - small contribution (indicates constraint but less informative)
literal_specificity(\+(Goal), Score) :-
    !,
    (   Goal = matches(_, Category) ->
        category_specificity(Category, BaseScore),
        Score is BaseScore * 0.3  % Negations worth less than positive matches
    ;   Score = 3
    ).

literal_specificity(\\+(Goal), Score) :-
    !,
    literal_specificity(\+(Goal), Score).

% Variable value check (e.g., query_intent(prevention))
% Check if it's a known variable from rule_info
literal_specificity(Literal, Score) :-
    Literal =.. [Functor, _Value],
    % Check if this functor is a known variable (appears as a rule head)
    rule_info(_, Goal, _, _),
    Goal =.. [Functor|_],
    !,
    Score = 5.

% Default case - minimal specificity
literal_specificity(_, 1).

/**
 * category_specificity(+Category, -Score)
 * Calculate specificity for a semantic category
 *
 * Uses inverse of category size: 1000 / num_patterns
 * Smaller categories (more specific terms) get higher scores
 *
 * Can be overridden with explicit category_weight/2 facts
 */
category_specificity(Category, Score) :-
    % Check for explicit weight first
    (   category_weight(Category, ExplicitWeight) ->
        Score = ExplicitWeight
    ;   % Calculate from pattern count
        findall(P, pattern(Category, P), Patterns),
        length(Patterns, NumPatterns),
        (   NumPatterns > 0 ->
            % Inverse relationship: fewer patterns = higher specificity
            Score is 1000 / NumPatterns
        ;   % No patterns defined - use moderate default
            Score = 50
        )
    ).

/**
 * explain_specificity(+BodyList, -Explanation)
 * Generate human-readable explanation of specificity calculation
 */
explain_specificity([], []).
explain_specificity([Lit|Rest], [LitExpl|RestExpl]) :-
    literal_specificity(Lit, Score),
    term_to_atom(Lit, LitAtom),
    format(atom(LitExpl), '~w: ~2f', [LitAtom, Score]),
    explain_specificity(Rest, RestExpl).

/**
 * properly_defeats(+Arg1, +Arg2)
 * True if Arg1 properly defeats Arg2 (not just blocks)
 */
properly_defeats(Arg1, Arg2) :-
    Arg1 = argument(_, RuleId1, _, _),
    Arg2 = argument(_, RuleId2, _, _),
    contradicts_argument(Arg1, Arg2),
    (   sup(RuleId1, RuleId2) ->
        true
    ;   \+ sup(RuleId2, RuleId1),
        is_more_specific(Arg1, Arg2)
    ).

/**
 * blocks(+Arg1, +Arg2)
 * True if Arg1 and Arg2 block each other (neither is preferred)
 */
blocks(Arg1, Arg2) :-
    Arg1 = argument(_, RuleId1, _, _),
    Arg2 = argument(_, RuleId2, _, _),
    contradicts_argument(Arg1, Arg2),
    \+ sup(RuleId1, RuleId2),
    \+ sup(RuleId2, RuleId1),
    \+ is_more_specific(Arg1, Arg2),
    \+ is_more_specific(Arg2, Arg1).

/**
 * contradicts_argument(+Arg1, +Arg2)
 * Two arguments are in conflict if their conclusions contradict
 */
contradicts_argument(argument(Goal1, _, _, _), argument(Goal2, _, _, _)) :-
    contradicts(Goal1, Goal2).

/**
 * contradicts(+Goal1, +Goal2)
 * True if Goal1 and Goal2 are contradictory
 * For this DSL: same functor/arity, but not identical terms
 */
contradicts(Goal1, Goal2) :-
    functor(Goal1, Functor, Arity),
    functor(Goal2, Functor, Arity),
    Arity > 0,
    Goal1 \= Goal2.

%% ============================================================================
%% 4. Dialectical Tree Construction
%% ============================================================================

/**
 * build_all_trees(+Arguments, -Trees)
 * Build dialectical trees for all arguments
 */
build_all_trees([], []).
build_all_trees([Arg|Args], [Tree|Trees]) :-
    build_dialectical_tree(Arg, Tree, 0, []),
    build_all_trees(Args, Trees).

/**
 * build_dialectical_tree(+Argument, -Tree, +Depth, +AncestorArgs)
 * Build a dialectical tree with Argument as root
 *
 * AncestorArgs: Arguments in the path from root (for acceptability check)
 */
build_dialectical_tree(Argument, tree(Argument, unmarked, DefeatersTree), Depth, Ancestors) :-
    Depth < 10,  % Depth limit
    !,
    % Trace: starting to build tree for this argument
    (Depth =:= 0 -> trace_tree_build_start(Argument) ; true),
    % Check concordance: defeater must not contradict ancestors
    findall(Defeater,
            (   find_single_defeater(Argument, Defeater),
                is_acceptable_defeater(Defeater, Ancestors)
            ),
            Defeaters),
    % Trace: defeaters found
    trace_defeaters(Argument, Defeaters),
    Depth1 is Depth + 1,
    NewAncestors = [Argument|Ancestors],
    build_defeaters_trees(Defeaters, DefeatersTree, Depth1, NewAncestors).

% Helper to trace defeaters
trace_defeaters(_, []).
trace_defeaters(Argument, [Defeater|Rest]) :-
    determine_defeat_type(Defeater, Argument, DefeatType),
    trace_defeater_found(Argument, Defeater, DefeatType),
    trace_defeaters(Argument, Rest).

% Determine defeat type (proper or blocking)
determine_defeat_type(Defeater, Argument, DefeatType) :-
    (   properly_defeats(Defeater, Argument)
    ->  DefeatType = proper_defeat
    ;   DefeatType = blocking_defeat
    ).

build_dialectical_tree(Argument, tree(Argument, undecided, []), _, _) :-
    % Depth limit reached
    !.

/**
 * is_acceptable_defeater(+Defeater, +Ancestors)
 * A defeater is acceptable if it doesn't contradict any ancestor
 * (Prevents circular argumentation)
 */
is_acceptable_defeater(Defeater, Ancestors) :-
    Defeater = argument(DefGoal, _, _, _),
    \+ (member(Ancestor, Ancestors),
        Ancestor = argument(AncGoal, _, _, _),
        contradicts(DefGoal, AncGoal)).

/**
 * build_defeaters_trees(+Defeaters, -Trees, +Depth, +Ancestors)
 * Build trees for all defeater arguments
 */
build_defeaters_trees([], [], _, _).
build_defeaters_trees([Defeater|Rest], [Tree|Trees], Depth, Ancestors) :-
    build_dialectical_tree(Defeater, Tree, Depth, Ancestors),
    build_defeaters_trees(Rest, Trees, Depth, Ancestors).

%% ============================================================================
%% 5. Marking Algorithm (Proper DeLP)
%% ============================================================================

/**
 * mark_trees(+Trees, -MarkedTrees)
 * Mark all trees with status
 */
mark_trees([], []).
mark_trees([Tree|Trees], [MarkedTree|MarkedTrees]) :-
    mark_tree(Tree, MarkedTree),
    mark_trees(Trees, MarkedTrees).

/**
 * mark_tree(+Tree, -MarkedTree)
 * Mark a single dialectical tree (bottom-up)
 *
 * Status values:
 *   - undefeated (U): No defeaters, or all defeaters are defeated
 *   - defeated (D): Has at least one undefeated proper defeater
 *   - blocked (B): Has blocking defeaters with no resolution
 *   - undecided: Has undecided defeaters
 */
mark_tree(tree(Arg, undecided, []), tree(Arg, undecided, [])) :-
    % Depth-limited node: preserve undecided status
    !,
    trace_marking(Arg, undecided, depth_limit_reached).

mark_tree(tree(Arg, _, []), tree(Arg, undefeated, [])) :-
    % Leaf node with no defeaters → undefeated
    !,
    trace_marking(Arg, undefeated, no_defeaters).

mark_tree(tree(Arg, _, DefeatersTree), tree(Arg, Status, MarkedDefeatersTree)) :-
    % Mark all defeater subtrees first (bottom-up)
    mark_trees(DefeatersTree, MarkedDefeatersTree),
    % Compute status based on defeaters
    compute_node_status(Arg, MarkedDefeatersTree, Status, Reason),
    trace_marking(Arg, Status, Reason).

/**
 * compute_node_status(+Argument, +MarkedDefeaters, -Status, -Reason)
 * Compute status considering proper defeats vs blocking
 * Also returns the reason for the status
 */
compute_node_status(Arg, MarkedDefeaters, Status, Reason) :-
    % Separate defeaters by type
    partition_defeaters(Arg, MarkedDefeaters, ProperDefeaters, BlockingDefeaters),

    % Check for undefeated proper defeaters
    (   has_undefeated_proper_defeater(ProperDefeaters) ->
        Status = defeated,
        find_undefeated_defeater(ProperDefeaters, DefeaterRule),
        Reason = defeated_by(DefeaterRule)
    ;   % Check for undefeated blocking defeaters
        has_undefeated_blocker(BlockingDefeaters) ->
        Status = blocked,
        find_undefeated_defeater(BlockingDefeaters, BlockerRule),
        Reason = blocked_by(BlockerRule)
    ;   % Check for undecided defeaters
        member(tree(_, undecided, _), MarkedDefeaters) ->
        Status = undecided,
        Reason = has_undecided_defeaters
    ;   % All defeaters are defeated → undefeated
        Status = undefeated,
        Reason = all_defeaters_defeated
    ).

% Find the rule ID of an undefeated defeater
find_undefeated_defeater(Defeaters, RuleId) :-
    member(tree(argument(_, RuleId, _, _), undefeated, _), Defeaters),
    !.
find_undefeated_defeater(_, unknown).

/**
 * partition_defeaters(+Arg, +Defeaters, -ProperDefeaters, -BlockingDefeaters)
 * Separate defeaters into proper defeaters and blocking defeaters
 */
partition_defeaters(_, [], [], []).
partition_defeaters(Arg, [tree(DefArg, Status, Children)|Rest],
                    ProperDefeaters, BlockingDefeaters) :-
    partition_defeaters(Arg, Rest, RestProper, RestBlocking),
    (   properly_defeats(DefArg, Arg) ->
        ProperDefeaters = [tree(DefArg, Status, Children)|RestProper],
        BlockingDefeaters = RestBlocking
    ;   ProperDefeaters = RestProper,
        BlockingDefeaters = [tree(DefArg, Status, Children)|RestBlocking]
    ).

/**
 * has_undefeated_proper_defeater(+ProperDefeaters)
 * Check if any proper defeater is undefeated
 */
has_undefeated_proper_defeater(ProperDefeaters) :-
    member(tree(_, undefeated, _), ProperDefeaters).

/**
 * has_undefeated_blocker(+BlockingDefeaters)
 * Check if any blocking defeater is undefeated (causes blocking)
 */
has_undefeated_blocker(BlockingDefeaters) :-
    member(tree(_, undefeated, _), BlockingDefeaters).

%% ============================================================================
%% 6. Tree Serialization for Janus
%% ============================================================================

/**
 * tree_to_dict(+Tree, -Dict)
 * Convert dialectical tree to dict format for Janus
 */
tree_to_dict(no_arguments(Goal), Dict) :-
    !,
    term_to_atom(Goal, GoalAtom),
    Dict = _{type: no_arguments, goal: GoalAtom}.

tree_to_dict(tree(Argument, Status, Defeaters), Dict) :-
    argument_to_dict(Argument, ArgDict),
    maplist(tree_to_dict, Defeaters, DefeatersDict),
    Dict = _{
        type: tree,
        argument: ArgDict,
        status: Status,
        defeaters: DefeatersDict
    }.

/**
 * argument_to_dict(+Argument, -Dict)
 * Convert argument to dict format
 */
argument_to_dict(argument(Goal, RuleId, Premises, SpecificityScore), Dict) :-
    Goal =.. [Functor|Args],
    maplist(safe_term_to_atom, Args, ArgsAtoms),
    maplist(premise_to_dict, Premises, PremisesDict),
    Dict = _{
        goal: Functor,
        goal_args: ArgsAtoms,
        rule_id: RuleId,
        premises: PremisesDict,
        specificity: SpecificityScore
    }.

/**
 * premise_to_dict(+Premise, -Dict)
 * Convert premise to dict format
 */
premise_to_dict(negation(Goal), Dict) :-
    !,
    term_to_atom(Goal, GoalAtom),
    Dict = _{type: negation, goal: GoalAtom, semantics: naf}.

premise_to_dict(not_warranted_premise(Goal), Dict) :-
    !,
    term_to_atom(Goal, GoalAtom),
    Dict = _{type: not_warranted, goal: GoalAtom, semantics: warrant_based}.

premise_to_dict(fact(Literal), Dict) :-
    !,
    Literal =.. [Functor|Args],
    maplist(safe_term_to_atom, Args, ArgsAtoms),
    Dict = _{type: fact, functor: Functor, args: ArgsAtoms}.

premise_to_dict(assumption(Literal), Dict) :-
    !,
    Literal =.. [Functor|Args],
    maplist(safe_term_to_atom, Args, ArgsAtoms),
    Dict = _{type: assumption, functor: Functor, args: ArgsAtoms}.

premise_to_dict(argument(Goal, RuleId, Premises, SpecificityScore), Dict) :-
    !,
    argument_to_dict(argument(Goal, RuleId, Premises, SpecificityScore), Dict).

/**
 * delp_query_dict(+Goal, -Status, -TreeDict)
 * Query with result tree as dict (Janus-friendly)
 */
delp_query_dict(Goal, Status, TreeDict) :-
    delp_query(Goal, Status, Tree),
    tree_to_dict(Tree, TreeDict).

%% ============================================================================
%% 7. Utility Predicates
%% ============================================================================

/**
 * safe_term_to_atom(+Term, -Atom)
 * Convert any term to an atom safely for Janus serialization.
 * Handles complex terms, unbound variables, and dicts.
 */
safe_term_to_atom(Term, Atom) :-
    (   var(Term)
    ->  Atom = '_'  % Unbound variable
    ;   atom(Term)
    ->  Atom = Term  % Already an atom
    ;   number(Term)
    ->  Atom = Term  % Numbers are fine
    ;   string(Term)
    ->  atom_string(Atom, Term)  % Convert string to atom
    ;   is_dict(Term)
    ->  term_to_atom(Term, Atom)  % Convert dict to atom representation
    ;   compound(Term)
    ->  term_to_atom(Term, Atom)  % Convert compound to atom
    ;   term_to_atom(Term, Atom)  % Fallback
    ).

/**
 * gensym(+Base, -Symbol)
 * Generate unique symbol
 */
gensym(Base, Symbol) :-
    (   retract(gensym_counter(Base, N)) ->
        N1 is N + 1
    ;   N1 = 1
    ),
    assertz(gensym_counter(Base, N1)),
    format(atom(Symbol), '~w~w', [Base, N1]).

/**
 * pretty_print_tree(+Tree)
 * Pretty print a dialectical tree for debugging
 */
pretty_print_tree(Tree) :-
    pretty_print_tree(Tree, 0).

pretty_print_tree(no_arguments(Goal), _) :-
    format('No arguments for ~w~n', [Goal]).

pretty_print_tree(tree(Arg, Status, Defeaters), Indent) :-
    indent(Indent),
    Arg = argument(Goal, RuleId, _, Specificity),
    format('~w [~w] via ~w (spec=~w)~n', [Goal, Status, RuleId, Specificity]),
    Indent1 is Indent + 2,
    pretty_print_defeaters(Defeaters, Indent1).

pretty_print_defeaters([], _).
pretty_print_defeaters([Tree|Rest], Indent) :-
    pretty_print_tree(Tree, Indent),
    pretty_print_defeaters(Rest, Indent).

indent(0) :- !.
indent(N) :-
    N > 0,
    write('  '),
    N1 is N - 1,
    indent(N1).

%% ============================================================================
%% 8. Debug/Inspection Predicates
%% ============================================================================

/**
 * list_all_arguments(+Goal, -Arguments)
 * List all arguments for a goal (for debugging)
 */
list_all_arguments(Goal, Arguments) :-
    findall(Arg, find_argument(Goal, Arg, []), Arguments).

/**
 * explain_defeat(+Arg1, +Arg2, -Explanation)
 * Explain why Arg1 defeats Arg2
 */
explain_defeat(Arg1, Arg2, Explanation) :-
    Arg1 = argument(_, RuleId1, _, Score1),
    Arg2 = argument(_, RuleId2, _, Score2),
    (   sup(RuleId1, RuleId2) ->
        Explanation = explicit_superiority(RuleId1, RuleId2)
    ;   Score1 > Score2 ->
        Explanation = more_specific(specificity_score(Score1), specificity_score(Score2))
    ;   Explanation = blocking
    ).

/**
 * compare_specificity(+Arg1, +Arg2, -Comparison)
 * Compare specificity of two arguments with detailed breakdown
 */
compare_specificity(Arg1, Arg2, Comparison) :-
    Arg1 = argument(_, RuleId1, _, Score1),
    Arg2 = argument(_, RuleId2, _, Score2),
    rule_info(RuleId1, _, _, Body1),
    rule_info(RuleId2, _, _, Body2),
    explain_specificity(Body1, Breakdown1),
    explain_specificity(Body2, Breakdown2),
    Comparison = comparison{
        rule1: RuleId1,
        score1: Score1,
        breakdown1: Breakdown1,
        rule2: RuleId2,
        score2: Score2,
        breakdown2: Breakdown2,
        winner: (Score1 > Score2 -> RuleId1 ; (Score2 > Score1 -> RuleId2 ; tie))
    }.

