include "global_cardinality_low_up.mzn";
include "global_cardinality.mzn";
include "regular.mzn";


array[TIME] of string: timeLabels;

int: ratio = 5;
% blocks
enum BLOCKS = {  ps, pp, pc, ls, lp, lc, bc, i, off };


int: maxTime;
int: max_required_live_classes_at_any_timeblock=  60;
set of int: TIME = 1..maxTime;  


array[1..card(BLOCKS)-2, TIME] of int: minNeeded;


set of int: SHIFT_ROWs = 1..max_required_live_classes_at_any_timeblock ;


%variables

array[SHIFT_ROWs, TIME] of var  BLOCKS: roster;


array[TIME] of var 0..(max_required_live_classes_at_any_timeblock): offBlocks;
%var 1..(max_required_live_classes_at_any_timeblock * 336): offBlocks;


%constraint offBlocks =  sum ( i in index_set( [roster[r,h] | r in SHIFT_ROWs, h in TIME ] )) ( bool2int(i in{ i, off })) ) ;
constraint forall(h in TIME) (offBlocks[h] =  sum ( j in index_set( [roster[r,h] | r in SHIFT_ROWs ] )) ( bool2int(j in { i, off })) ) ;
%constraint forall(h in TIME) (offBlocks[h] =  card (index_set( [roster[r,h] | r in SHIFT_ROWs where roster[r,h] in {off} ] )))  ;


 
constraint forall(h in TIME) ( global_cardinality([ 
roster[r, h] | r in SHIFT_ROWs],
[ ps, pp, pc, ls, lp, lc], 
[ ceil(minNeeded[ps,h]/ratio), ceil(minNeeded[pp,h]/ratio), ceil(minNeeded[pc,h]/ratio), ceil(minNeeded[ls,h]/ratio) , ceil(minNeeded[lp,h]/ratio), ceil(minNeeded[lc,h]/ratio)  ]
 ));
 



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

constraint forall(r in SHIFT_ROWs) (regular([roster[r,h] | h in TIME], Q, S, transition_array, q0, STEPS));                                   

solve :: int_search (        [ roster[r,h] | r in SHIFT_ROWs, h in TIME],         most_constrained,         indomain_min,        complete)     minimize sum(h in TIME) (offBlocks[h]);                          
                
%solve minimize sum(h in TIME) (offBlocks[h]);
output [show(roster)];

