%
% Prolog Helper Predicates for Canto
%
% These helpers are loaded once at startup to support syntax verification
% and other Prolog operations via Janus.
%

% read_all_terms_from_stream(+Stream)
% Read all terms from a stream until end_of_file
read_all_terms_from_stream(Stream) :-
    repeat,
    read_term(Stream, Term, []),
    (   Term == end_of_file
    ->  !
    ;   fail
    ).

% verify_prolog_syntax(+CodeString, -Result)
% Verify Prolog code syntax without executing it
% Result is 'true' if syntax is valid, or error(...) if invalid
verify_prolog_syntax(CodeString, Result) :-
    open_string(CodeString, Stream),
    catch(
        (
            read_all_terms_from_stream(Stream),
            close(Stream),
            Result = true
        ),
        Error,
        (
            close(Stream),
            Result = error(Error)
        )
    ).
