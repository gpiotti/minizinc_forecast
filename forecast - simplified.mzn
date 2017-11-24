include "global_cardinality_low_up.mzn";
include "global_cardinality.mzn";
include "at_most.mzn";
include "regular.mzn";

array[TIME] of string: timeLabels;
%array[1..card(BLOCK)] of string: blockName;

% blocks
enum BLOCK = {  p_1, p_2, p_3, p_4, l_1, l_2, l_3, bc_2, bc_3, bc_4, i_2, i_3, i_4, off };
        
enum BLOCKs = { p, l, o };

array[1..card(BLOCK)] of BLOCKs: block_to_blocks = [p,p,p,p,l,l,l,l,l,l,o,o,o,o];
                          

int: maxTime;
int: max_required_at_any_timeblock= 100;
set of int: TIME = 1..maxTime;  


array[BLOCKs, TIME] of int: minNeeded;

set of int: SHIFT_ROWs = 1..max_required_at_any_timeblock * 2;


%variables
array[SHIFT_ROWs, TIME] of var BLOCK: roster;


constraint forall(h in TIME) ( global_cardinality_low_up([ block_to_blocks[roster[r, h]] | r in SHIFT_ROWs],[p, l, o], [ minNeeded[p,h], minNeeded[l,h], minNeeded[o,h] ], [ max_required_at_any_timeblock * 2, max_required_at_any_timeblock * 2, max_required_at_any_timeblock * 2 ] ));
%  [p, l, o], [ minNeeded[p,h], minNeeded[l,h], minNeeded[o,h] ]  ));


int: Q = 15; %states quantity
int: S = 14; %input quantity
int: q0= 1; %starting state
set of int: STEPS = 1..Q;

array[1..Q, 1..S] of 0..Q: transition_array = 
[|2,0,0,0,6,0,0,0,0,0,0,0,0,15,
|0,3,0,0,0,7,0,0,0,0,12,0,0,0,
|0,0,4,0,0,0,8,0,0,0,0,13,0,0,
|0,0,0,5,0,0,0,0,0,0,0,0,14,0,
|2,0,0,0,6,0,0,0,0,0,0,0,0,15,
|0,0,0,0,0,0,0,9,0,0,0,0,0,0,
|0,0,0,0,0,0,0,0,10,0,0,0,0,0,
|0,0,0,0,0,0,0,0,0,11,0,0,0,0,
|0,0,4,0,0,0,8,0,0,0,0,13,0,0,
|0,0,0,5,0,0,0,0,0,0,0,0,14,0,
|2,0,0,0,6,0,0,0,0,0,0,0,0,15,
|0,0,4,0,0,0,8,0,0,0,0,13,0,0,
|0,0,0,5,0,0,0,0,0,0,0,0,14,0,
|2,0,0,0,6,0,0,0,0,0,0,0,0,15,
|2,0,0,0,6,0,0,0,0,0,0,0,0,15|];




constraint forall(r in SHIFT_ROWs) (regular([roster[r,h] | h in TIME], Q, S, transition_array, q0, STEPS));


                                         

                                         
                                          
solve satisfy;
output [show(roster)];
%output [ show(SHIFT_ROWs[i]) ++ ";" ++ timeLabels[j] ++ ";" ++ blockName[fix(roster[i,j])] ++ "\n"  | i in SHIFT_ROWs, j in TIME ]  ;