include "global_cardinality.mzn";
include "regular.mzn";


array[TIME] of string: timeLabels;

array[1..card(BLOCKS)-3] of int: ratios = [1,1,1,5,5,5];
% blocks
enum BLOCKS = {  ps, pp, pc, ls, lp, lc, bc, i, off }; 

set of int: real_blocks = {ps, pp, pc, ls, lp, lc};




int: maxTime;

set of int: TIME = 1..maxTime;
 


array[1..card(BLOCKS)-3, TIME] of int: minNeeded;


%set of int: SHIFT_ROWs = 1..max_required_live_classes_at_any_timeblock ;
set of int: SHIFT_ROWs = 1..maxRows ;

%variables

array[SHIFT_ROWs, TIME] of var  BLOCKS: roster ;


%array[SHIFT_ROWs] of var int: offBlocks ;
int: maxRows = 
  ceil(max([minNeeded[ps,h]| b in real_blocks, h in TIME])/ratios[ps]) +
  ceil(max([minNeeded[pp,h]| b in real_blocks, h in TIME])/ratios[pp]) +
  ceil(max([minNeeded[pc,h]| b in real_blocks, h in TIME])/ratios[pc]) +
  ceil(max([minNeeded[lc,h]| b in real_blocks, h in TIME])/ratios[lc]) * 2 + 
  ceil(max([minNeeded[lp,h]| b in real_blocks, h in TIME])/ratios[lp]) * 2 +
  ceil(max([minNeeded[ls,h]| b in real_blocks, h in TIME])/ratios[ls]) * 2;

var int: iBlocks ; 


constraint iBlocks =  sum (  [ bool2int(roster[r,h] == i) | h in TIME , r in SHIFT_ROWs  ]);
 
constraint forall(h in TIME) ( global_cardinality([ 
roster[r, h] | r in SHIFT_ROWs],
[ ps, pp, pc, ls, lp, lc], 
[ ceil(minNeeded[ps,h]/ratios[ps]), ceil(minNeeded[pp,h]/ratios[pp]), ceil(minNeeded[pc,h]/ratios[pc]), ceil(minNeeded[ls,h]/ratios[ls]) , ceil(minNeeded[lp,h]/ratios[lp]), ceil(minNeeded[lc,h]/ratios[lc])  ]
)) ;
 


int: Q = 44; %states quantity
int: S = 9; %input quantity
int: q0= 1; %starting state
set of int: STEPS = 1..Q;

array[1..Q, 1..S] of 0..Q: transition_array = 
[|2,3,4,5,6,7,0,0,1,
|8,0,12,13,0,18,0,23,0,
|0,9,11,0,14,17,0,22,0,
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
|40,41,42,0,0,0,0,44,0
|0,41,42,0,0,0,0,44,0,
|40,0,42,0,0,0,0,44,0,
|2,3,4,5,6,7,0,0,1,
|2,3,4,5,6,7,0,0,1,
|2,3,4,5,6,7,0,0,1,
|2,3,4,5,6,7,0,0,1,
|2,3,4,5,6,7,0,0,1|]
;

constraint forall(r in SHIFT_ROWs) (regular([roster[r,h] | h in TIME], Q, S, transition_array, q0, STEPS));                                   

solve :: int_search (  [  roster[r,h] | r in SHIFT_ROWs, h in TIME] ,   smallest,         indomain_min,        complete)     minimize iBlocks; 
%solve minimize iBlocks;
% output [show([roster[r,h] | r in SHIFT_ROWs, h in TIME] )] ++ ["\n"]  ++[show(offBlocks[2])]  ++ ["\n"] ++ [show([bool2int(roster[1,h] == off) | h in TIME] )] ;

output [ show(fix(roster[r,h]))  ++ "\t" ++ if h == maxTime then "\n" else "" endif | r in SHIFT_ROWs, h in TIME ] ++ ["\n"] ++ [show(iBlocks)] 


;