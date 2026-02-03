% Kobiety
% pierwszy rzad
kobieta(emilia).
kobieta(lucja).

% drugi rzad
kobieta(barbara).
kobieta(cecylia).
kobieta(ania).

% trzeci rzad
kobieta(genowefa).



% definicja malzenstw
malzenstwo(emilia, grzegorz).
malzenstwo(grzegorz, emilia).

malzenstwo(lucja, eustachy).
malzenstwo(eustachy, lucja).

malzenstwo(lukasz, ania).
malzenstwo(ania, lukasz).


% definicja dzieci
dziecko(kacper, janusz).
dziecko(kacper, emilia).

dziecko(barbara, grzegorz).
dziecko(barbara, emilia).

dziecko(lukasz, emilia).
dziecko(lukasz, grzegorz).

dziecko(marcin, lukasz).
dziecko(marcin, ania).

dziecko(genowefa, lukasz).
dziecko(genowefa, ania).

dziecko(michal, genowefa).
dziecko(michal, bartosz).

dziecko(cecylia, lucja).
dziecko(cecylia, eustachy).

dziecko(ania, lucja).
dziecko(ania, eustachy).



% Relacja dziadka: X to dziadek Y
dziadek(X, Y) :-
    dziecko(Z, X),
    dziecko(Y, Z),
    not(kobieta(X)).

% Sprawdzamy czy X jest dziadkiem kogokolwiek

% Sprawdzenie czy X jest na liscie Y
czy_jest_na_liscie(X, [X | _]). % powiedzenie ze zejezli X jest pierwszym elementem list Y to PRAWDA
czy_jest_na_liscie(X, [_ | Y]) :- % Przechodzimy przez liste
    czy_jest_na_liscie(X, Y).


% =========== LOGIKA DZIADKA
% Sprawdzamy czy istnieje jakis dziadek ktory nie jest zawarty w naszej
% liscie X
istnieje_brakujacy_dziadek(X) :-
    dziadek(Z, _),
    \+ czy_jest_na_liscie(Z, X).

% Sprawdzamy czy na naszej liscie same dziadki podajemy X i idziemyjest
% dalej, jezeli X nie bedzie jakimkolwiek dziadkiem to FALSZ
czy_na_liscie_same_dziadki([]).
czy_na_liscie_same_dziadki([X|Y]) :-
    dziadek(X, _),
    czy_na_liscie_same_dziadki(Y).

% Kobinacja wszystkich poprzednich funkcji? bardziej predykatow chyba
dziadek(X) :-
    czy_na_liscie_same_dziadki(X),
    \+ istnieje_brakujacy_dziadek(X).




% =========== LOGIKA PRADZIADKA
% Podobna procedura dla pradziadka
pradziadek(X, Y) :-
    dziadek(X, Z),
    dziecko(Y, Z).

istnieje_brakujacy_pradziadek(X) :-
    pradziadek(Z, _),
    \+ czy_jest_na_liscie(Z, X).

czy_na_liscie_same_pradziadki([]).
czy_na_liscie_same_pradziadki([X|Y]) :-
    pradziadek(X, _),
    czy_na_liscie_same_pradziadki(Y).

pradziadek(X) :-
    czy_na_liscie_same_pradziadki(X),
    \+ istnieje_brakujacy_pradziadek(X).





% =========== Logika siostry
% Wszystkie te relacje beda bardzo podobne roznic bedzie sie jedynie
% definicja K(X, Y)

% X jest siostra Y kiedy oboje posiadaja choc jednego tego samego
% rodzica i X jest kobieta
siostra(X, Y) :-
    dziecko(X, Z), dziecko(Y, Z),
    kobieta(X).

istnieje_brakujaca_siostra(X) :-
    siostra(Z, _),
    \+ czy_jest_na_liscie(Z, X).

czy_na_liscie_same_siostry([]).
czy_na_liscie_same_siostry([X | Y]) :-
    siostra(X, _),
    czy_na_liscie_same_siostry(Y).

siostra(X) :-
    czy_na_liscie_same_siostry(X),
    \+ istnieje_brakujaca_siostra(X).


% ========= Logika brata
brat(X, Y) :-
    dziecko(X, Z), dziecko(Y, Z),
    not(kobieta(X)).

istnieja_brakujacy_bracia(X) :-
    brat(Z, _),
    \+ czy_jest_na_liscie(Z, X).

czy_na_liscie_sami_bracia([]).
czy_na_liscie_sami_bracia([X | Y]) :-
    siostra(X, _),
    czy_na_liscie_same_siostry(Y).

brat(X) :-
    czy_na_liscie_sami_bracia(X),
    \+ istnieja_brakujacy_bracia(X).



% LOGIKA ZWRACAJACA POTOMKOW
potomek(X, Y) :- dziecko(X, Y).
potomek(X, Y) :-
    dziecko(X, Z),
    potomek(Z, Y).

% ciekawe jak zamienimy kolejnosc i zdefiniujemy
% znajdz_potomkow(_, Y, Y). to prolog czytajac program od gory zwroci
% odrazu W = []

znajdz_potomkow(X, Y, W) :-
    potomek(Z, X),
    \+ czy_jest_na_liscie(Z, Y),
    !,
    znajdz_potomkow(X, [Z | Y], W).

znajdz_potomkow(_, Y, Y).


% LOGIKA ZWRACAJA PRZODKOW
przodek(X, Y) :- dziecko(Y, X).
przodek(X, Y) :-
    dziecko(Y, Z),
    przodek(X, Z).

znajdz_przodkow(X, Y, W) :-
    przodek(Z, X),
    \+ czy_jest_na_liscie(Z, Y),
    !,
    znajdz_przodkow(X, [Z | Y], W).

znajdz_przodkow(_, Y, Y).

% Zliczenie Ile elementow w liscie
list_len([], 0).
list_len([_ | Y], W) :-
    list_len(Y, N),
    W is N + 1.


ile_przodkow(X, W) :-
    znajdz_przodkow(X, [], Z),
    list_len(Z, W).


ile_potomkow(X, W) :-
    znajdz_potomkow(X, [], Z),
    list_len(Z, W).

okresl_relacje(X, Y,'syn'):- dziecko(Y, X), not(kobieta(Y)).
okresl_relacje(X, Y,'corka'):- dziecko(Y, X), kobieta(Y).

okresl_relacje(X, Y,'wnuk'):- dziadek(X, Y), not(kobieta(Y)).
okresl_relacje(X, Y,'wnuczka'):- dziadek(X, Y), kobieta(Y).

okresl_relacje(X, Y, 'prawnuk'):- pradziadek(Y, X), not(kobieta(Y)).
okresl_relacje(X, Y, 'prawnuczka'):- pradziadek(Y, X), kobieta(Y).
okresl_relacje(_, _, 'potomek').

format_pary(X,[Z|Y], S) :-
    length(Y, 0),
    okresl_relacje(X, Z, B),
    format(atom(S), '~w(~w)', [Z, B]).

format_pary(X, [Z| Y], S) :-
    not(length(Y, 0)),
    format_pary(X, Y, V),
    okresl_relacje(X, Z, Q),
    format(atom(S), '~w(~w) | ~w', [Z,Q, V]).




info(X) :-
    znajdz_potomkow(X, [], W),
    format_pary(X, W, S),
    format('~w', [S]).



