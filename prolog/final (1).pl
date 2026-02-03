
% ========================================================= Zadanie 4.1
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
kobieta(julia).
kobieta(alicja).


% definicja malzenstw
malzenstwo(emilia, grzegorz).
malzenstwo(grzegorz, emilia).

malzenstwo(lucja, eustachy).
malzenstwo(eustachy, lucja).

malzenstwo(lukasz, ania).
malzenstwo(ania, lukasz).

malzenstwo(marcin, julia).
malzenstwo(julia, marcin).

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

dziecko(alicja, lukasz).
dziecko(alicja, ania).


% ========================================================= Zadanie 4.2

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
    brat(X, _),
    czy_na_liscie_sami_bracia(Y).

brat(X) :-
    czy_na_liscie_sami_bracia(X),
    \+ istnieja_brakujacy_bracia(X).


% ========================================================= Zadanie 4.3

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
% ========================================================= Zadanie 4.4

okresl_relacje(X, Y,'syn'):- dziecko(Y, X), not(kobieta(Y)).
okresl_relacje(X, Y,'corka'):- dziecko(Y, X), kobieta(Y).

okresl_relacje(X, Y,'wnuk'):- dziadek(X, Y), not(kobieta(Y)).
okresl_relacje(X, Y,'wnuczka'):- dziadek(X, Y), kobieta(Y).

okresl_relacje(X, Y, 'prawnuk'):- pradziadek(X, Y), not(kobieta(X)).
okresl_relacje(X, Y, 'prawnuczka'):- pradziadek(X, Y), kobieta(X).
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
% ========================================================= Zadanie 4.5

% Logika zwracająca osoby bez wstępnych i bez zstępnych
bezzstepne(X) :-
    (dziecko(X, _); malzenstwo(X, _); malzenstwo(_, X)),
    \+ dziecko(_, X).

bezwstepne(X) :-
    (dziecko(_, X); malzenstwo(X, _); malzenstwo(_, X)),
    \+ dziecko(X, _).

% Logika zwracająca listy osób bez wstępnych i bez zstępnych
znajdz_bezzstepne(Y, W) :-
    bezzstepne(Z),
    \+ czy_jest_na_liscie(Z, Y),
    znajdz_bezzstepne([Z | Y], W).

znajdz_bezzstepne(W, W).

znajdz_bezwstepne(Y, W) :-
	bezwstepne(Z),
	\+ czy_jest_na_liscie(Z, Y),
	znajdz_bezwstepne([Z | Y], W).

znajdz_bezwstepne(W, W).

% Zwraca liczbe osob bez wstepnych i bez stepnych
ile_bezwstepnych(N) :-
	znajdz_bezwstepne([], L),
    list_len(L, N).

ile_bezzstepnych(N) :-
	znajdz_bezzstepne([], L),
    list_len(L, N).

% Ponieważ chcemy obliczyć ilość siostr bez korzystania listy
% musimy poslozyc sie sprawdzaniem molziwych powiazan z wykorzystaniem
% alfabetycznego ulozenia atomow, bedziemy przechodzic po kolei
% alfabetycznie co zapewni nas ze nie zapetlimy sie

ile_dodaj(X, Y, 2) :-
    siostra(X, Y).

ile_dodaj(_,_, 1).

% inspiracaj do rozwiazania tego zadania:
% https://www.cs.gordon.edu/courses/cs323/PROLOG/prolog.html
czy_posortowana([X, Y | Z]) :-
    X @=< Y,
    czy_posortowana([Y | Z]).
czy_posortowana([_]).

% Glowna inspiracja do rozwiazania problemu
% https://www.cs.gordon.edu/courses/cs323/PROLOG/prolog.htmlc Atom w
% prologu jest tak jakby identyfikatorem obiektu, np. alicja lub
% 'Alicja' ale Alicja jest zmienna "Alicja" jest stringiem

sprawdzenie_kolejnosci(X, Y, W) :- % X Y to ostatnio odwiedzona para
    siostra(Z, N), % szukamy w bazie wiedzy Z, N gdzie Z jest siostra N
    (Z @> X ; (Z == X, N @> Y)),% sprawdzenie czy nowa para jest alfabetycznie dalej jezeli jezeli X = 10 to Z muzi byc conajmniej 11 jak wpiszemy atom_codes() to mozemy zobaczyc te wartosci
    Z @< N, % usuniecie powtorzen jak mamy alicja - genowefa to nie prawdzamy genowefa - alicja
    % chcemy upewnic sie ze nie istnieje taka para siostr ktora jest blizej pary X, Y. (X, Y) < (K, L) < (Z,N) <- i chcemy zeby tego nie znalazl
    \+ (
             siostra(K, L),
             K @< L,
             (Z @> K ; (Z == K , N @> L)),
             (K @> X ; (K == X , L @> Y))
       ),
    !,
    sprawdzenie_kolejnosci(Z, N, WK),
    ile_dodaj(N, Z, B),
    W is WK + B.

sprawdzenie_kolejnosci(_, _, 0).

policz_siostry(W) :-
    sprawdzenie_kolejnosci(0, 0, W).
