include "global_cardinality_low_up.mzn";
include "global_cardinality.mzn";
include "at_most.mzn";
include "regular.mzn";
include "regular_nfa.mzn";
include "count_lt.mzn";
include "inverse.mzn";

array[TIME] of string: timeLabels;


% blocks
enum BLOCK_BY_POSITION = {  p_1, p_2, p_3, p_4, l_1, l_2, l_3, bc_2, bc_3, bc_4, i_2, i_3, i_4, off };

        
enum BLOCK_BY_TYPE = { p, l, o };

enum BLOCK_BY_COURSE = { pp, ps, pc, lp, ls, lc, oo };

set of BLOCK_BY_COURSE: portugueseShifts = {pp,lp};
set of BLOCK_BY_COURSE: spanishShifts = {ps,ls};

set of BLOCK_BY_POSITION: SHIFT_INIT_BLOCK = { p_1, l_1 };

array[1..card(BLOCK_BY_COURSE)] of int: values = [1,1,1,1,1,1,0];


array[BLOCK_BY_COURSE, BLOCK_BY_POSITION] of bool : compatible; 
array[BLOCK_BY_POSITION, BLOCK_BY_COURSE] of bool : compatible_inv; 
                          

int: maxTime;
int: max_required_live_classes_at_any_timeblock=  10;
set of int: TIME = 1..maxTime;  


array[BLOCK_BY_COURSE, TIME] of int: minNeeded;


set of int: SHIFT_ROWs = 1..max_required_live_classes_at_any_timeblock ;


%variables
array[SHIFT_ROWs, TIME] of var  BLOCK_BY_POSITION: roster_position;
array[SHIFT_ROWs, TIME] of var  BLOCK_BY_COURSE: roster;


%constraint inverse(position_to_type[roster_position], position_to_type[roster]);
constraint forall(r in SHIFT_ROWs, h in TIME) ( compatible[ roster[r,h], roster_position[r,h]] == true
 );

% constraint forall(r in SHIFT_ROWs, h in TIME) (
% compatible[ roster[r,h], roster_position[r,h]] == compatible_inv[ roster_position[r,h], roster[r,h]] 
% <->
% compatible_inv[ roster_position[r,h], roster[r,h]] == compatible[ roster[r,h], roster_position[r,h]]


% );


var 0..max_required_live_classes_at_any_timeblock: offBlocks;


constraint offBlocks =  sum ( i in index_set( [roster_position[r,h] | r in SHIFT_ROWs, h in TIME ] )) ( bool2int(i in{ i_2, i_3, i_4})) 
+
 sum ( i in index_set( [roster[r,h] | r in SHIFT_ROWs, h in TIME ] )) ( bool2int(i in{ oo})) ;


% constraint forall(h in TIME) ( global_cardinality_low_up([ 
% roster_position[r, h] | r in SHIFT_ROWs],
% [ p_1, p_2, p_3, p_4, l_1, l_2, l_3], 
% [ max([minNeeded[pp,h], minNeeded[ps,h] , minNeeded[pc,h]]),
% max( [minNeeded[pp,h] , minNeeded[ps,h] , minNeeded[pc,h]]),
% max( [minNeeded[pp,h] , minNeeded[ps,h] , minNeeded[pc,h]]),
% max( [minNeeded[pp,h] , minNeeded[ps,h] , minNeeded[pc,h]]),
% max( [minNeeded[lp,h] , minNeeded[ls,h], minNeeded[lc,h]]),
% max( [minNeeded[lp,h] , minNeeded[ls,h], minNeeded[lc,h]]),
% max( [minNeeded[lp,h] , minNeeded[ls,h], minNeeded[lc,h]]) ]
%  ,
%  [max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock,
%  max_required_live_classes_at_any_timeblock] 
%  ));
 
 
 
constraint forall(h in TIME) ( global_cardinality([ 
roster[r, h] | r in SHIFT_ROWs],
[ pp, ps, pc, lp, ls, lc], 
[ minNeeded[pp,h], minNeeded[ps,h], minNeeded[pc,h], minNeeded[lp,h] , minNeeded[ls,h], minNeeded[lc,h]  ]
 ));
 


constraint forall (r in SHIFT_ROWs, h in 1..maxTime-3) (
roster_position[r,h] in SHIFT_INIT_BLOCK /\ roster[r,h] in portugueseShifts ->
not exists(  i in h..+3 )( roster[r,i] in spanishShifts )); 

constraint forall (r in SHIFT_ROWs, h in 1..maxTime-3) (
roster_position[r,h] in SHIFT_INIT_BLOCK /\ roster[r,h] in spanishShifts ->
not exists(  i in h..+3 )( roster[r,i] in portugueseShifts )); 



int: Q = 45; %states quantity
int: S = 9; %input quantity
int: q0= 1; %starting state
set of int: STEPS = 1..Q;

array[1..Q, 1..S] of 0..Q: transition_array = 
[|2,3,4,5,6,7,0,0,45,
|8,0,12,13,0,18,0,23,0,
|0,9,11,0,2,17,0,22,0,
|8,9,10,2,14,15,0,21,0,
|0,0,0,0,0,0,20,0,0,
|0,0,0,0,0,0,19,0,0,
|0,0,0,0,0,0,16,0,0,
|24,0,28,29,0,34,0,39,0,
|0,25,27,0,30,33,0,38,0,
|24,25,26,29,30,31,0,37,0,
|0,25,27,0,30,33,0,38,0,
|24,0,28,29,0,34,0,39,0,
|0,0,0,0,0,0,36,0,0,
|0,0,0,0,0,0,35,0,0,
|0,0,0,0,0,0,32,0,0,
|24,25,26,29,30,31,0,37,0,
|0,0,0,0,0,0,35,0,0,
|0,0,0,0,0,0,36,0,0,
|0,25,27,0,30,33,0,38,0,
|24,0,28,29,0,34,0,39,0,
|24,25,26,29,30,31,0,37,0,
|0,25,27,0,30,33,0,38,0,
|24,0,28,29,0,34,0,39,0,
|40,0,42,0,0,0,0,44,0,
|0,41,0,0,0,0,0,44,0,
|40,41,42,0,0,0,0,44,0,
|0,41,42,0,0,0,0,44,0,
|40,0,42,0,0,0,0,44,0,
|0,0,0,0,0,0,43,0,0,
|0,0,0,0,0,0,43,0,0,
|0,0,0,0,0,0,43,0,0,
|40,41,42,0,0,0,44,0,0,
|0,0,0,0,0,0,43,0,0,
|0,0,0,0,0,0,43,0,0,
|0,41,42,0,0,0,0,44,0,
|40,0,42,0,0,0,0,44,0,
|40,41,42,0,0,0,0,44,0,
|0,41,42,0,0,0,0,44,0,
|40,0,42,0,0,0,0,44,0,
|2,3,4,5,6,7,0,0,45,
|2,3,4,5,6,7,0,0,45,
|2,3,4,5,6,7,0,0,45,
|2,3,4,5,6,7,0,0,45,
|2,3,4,5,6,7,0,0,45,
|2,3,4,5,6,7,0,0,45|]

;

constraint forall(r in SHIFT_ROWs) (regular([roster_position[r,h] | h in TIME], Q, S, transition_array, q0, STEPS));                                   

             solve :: int_search (        [ roster_position[r,h] | r in SHIFT_ROWs, h in TIME],         most_constrained,         indomain_min,        complete)     minimize offBlocks;                          
%solve minimize offBlocks;
%output [show(roster) ++ show(roster_position)];
output [show(roster)]  ++ ["\n"] ++ [show(roster_position)];

%output [ show(SHIFT_ROWs[i]) ++ ";" ++ timeLabels[j] ++ ";" ++ blockName[fix(roster[i,j])] ++ "\n"  | i in SHIFT_ROWs, j in TIME ]  ;