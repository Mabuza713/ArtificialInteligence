% dziadkowie
mezczyzna(eustachy).
mezczyzna(janusz).
mezczyzna(norbert).

kobieta(lubmila).
kobieta(anna).
kobieta(bazylia).
kobieta(paulina).


% Rodzice
mezczyzna(eugeniusz).
mezczyzna(piotr).

kobieta(genowefa).
kobieta(grazyna).


% Dzieci
mezczyzna(michal).
mezczyzna(maciej).

kobieta(kasia).
kobieta(asia).


% relacja matczyna i ojcowska dla dziadkow i rodzicow matka lub ojciec rodzic(X, Y) gdzy X jest rodzicem Y
rodzic(eustachy, eugeniusz).
rodzic(lubmila, eugeniusz).

rodzic(janusz, genowefa).
rodzic(paulina, genowefa).

rodzic(janusz, piotr).
rodzic(anna, piotr).

rodzic(norbert, grazyna).
rodzic(bazylia, grazyna).


% relacja matczyna i ojcowska dla rodzicow i dzieci
rodzic(eugeniusz, kasia).
rodzic(eugeniusz, asia).
rodzic(eugeniusz, michal).
rodzic(genowefa, kasia).
rodzic(genowefa, asia).
rodzic(genowefa, michal).

rodzic(piotr, maciej).
rodzic(grazyna, maciej).

rodzic(ezykiel, eugeniusz).
rodzic(ezykiel, augiasz).

rodzic(ezykiel, dagmara).

% definicja syna i corki
syn(X, Y) :- (rodzic(Y, X), mezczyzna(X)).
corka(X, Y) :- (rodzic(Y, X), kobieta(X)).

% definicja ojca i matki
ojciec(X, Y) :- rodzic(X, Y), mezczyzna(X).
matka(X, Y) :- rodzic(X, Y), kobieta(X).

% definicja dziadka i babci X jest dziadkiem Y
dziadek(X, Y) :- ojciec(X, T) , (ojciec(T, Y) ; matka(T, Y)).
babcia(X, Y) :- matka(X, T), (ojciec(T, Y); matka(T,  Y)).

% definicja rodzensta
rodzenstwo_conajmniej_przybrane(X, Y) :-((ojciec(T, X) , ojciec(T,Y)) ; (matka(T, X), matka(T, Y))), X \= Y.

brat(X, Y) :- rodzenstwo_conajmniej_przybrane(X, Y) , mezczyzna(X).
siostra(X, Y) :- rodzenstwo_conajmniej_przybrane(X, Y), kobieta(X).

wnuk(X, Y) :- (dziadek(Y, X) ; babcia(Y, X)) , mezczyzna(X)
wnuczka(X, Y) :- (dziadek(Y, X) ; babcia(Y, X)) , kobieta(X)


malzenstwo(X, Y) :- rodzic(X, Z) , rodzic(Y, Z).

czy_rodzina(X, Y) :- 
	ojciec(X, Y);
	malzenstwo(X, Y);
	matka(X,Y);
	syn(X,Y);
	corka(X,Y).





przodek(X, Y) :- rodzic(X, Y).
przodek(X, Y) :-
	rodzic(X, Z),
	przodek(Z, Y).

% kuzyni i dalsza rodzina np. michal z maciejem, poprzez dziadka janusza
maja_wspolnego_przodka(X,Y) :-
	przodek(Z, X),
	przodek(Z, Y).

potomek(X, Y) :- rodzic(Y, X).
potomek(X, Y) :-
	rodzic(Z, X),
	potomek(Z, Y).

func(A, B, A) :- A <= B, !. func(_, B, B).

czy_ta_sama_krew(X, Y) :-
	potomek(X,Y) ;
	przodek(X, Y);
	maja_wspolnego_przodka(X, Y);
	rodzenstwo_conajmniej_przybrane(X, Y).