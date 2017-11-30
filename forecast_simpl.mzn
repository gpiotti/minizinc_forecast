include "global_cardinality_low_up.mzn";
include "global_cardinality.mzn";
include "at_most.mzn";
include "regular.mzn";
include "regular_nfa.mzn";
include "count_lt.mzn";

array[TIME] of string: timeLabels;


% blocks
enum BLOCK_BY_POSITION = {  p_1, p_2, p_3, p_4, l_1, l_2, l_3, bc_2, bc_3, bc_4, i_2, i_3, i_4, off };
        
enum BLOCK_BY_TYPE = { p, l, o };

enum BLOCK_BY_COURSE = { pp, ps, pc, lp, ls, lc, oo };

set of BLOCK_BY_COURSE: portugueseShifts = {pp,lp,oo};
set of BLOCK_BY_COURSE: spanishShifts = {ps,ls,oo};

set of BLOCK_BY_POSITION: SHIFT_INIT_BLOCK = { p_1, l_1 };

array[1..card(BLOCK_BY_COURSE)] of int: values = [1,1,1,1,1,1,0];

array[1..card(BLOCK_BY_POSITION)] of BLOCK_BY_TYPE: position_to_type = [p,p,p,p,l,l,l,o,o,o,o,o,o,o];
array[1..card(BLOCK_BY_COURSE)] of BLOCK_BY_TYPE: course_to_type = [p,p,p,l,l,l,o];

                          

int: maxTime;
int: max_required_live_classes_at_any_timeblock=  100;
set of int: TIME = 1..maxTime;  


array[BLOCK_BY_COURSE, TIME] of int: minNeeded;


set of int: SHIFT_ROWs = 1..max_required_live_classes_at_any_timeblock ;


%variables
array[SHIFT_ROWs, TIME] of var BLOCK_BY_POSITION: roster_position;
array[SHIFT_ROWs, TIME] of var BLOCK_BY_COURSE: roster;

constraint forall(r in SHIFT_ROWs, h in TIME) ( course_to_type[roster[r,h]] == position_to_type[roster_position[r,h]] );

array[ {lp,ls,lc}, 1..maxTime] of var int: continuedBlocks;

constraint forall (s in {lp, ls, lc}) ( continuedBlocks[s,1] = 0 );
constraint forall (h in 2..maxTime, s in {lp, ls, lc}) ( continuedBlocks[s,h] =  minNeeded[s,h-1] );
 
 
 
constraint forall(h in TIME) ( global_cardinality([ 
roster[r, h] | r in SHIFT_ROWs],
[ pp, ps, pc, lp, ls, lc], 
[ minNeeded[pp,h], minNeeded[ps,h], minNeeded[pc,h], minNeeded[lp,h] , minNeeded[ls,h], minNeeded[lc,h]  ]
 ));
 
%  constraint forall(h in 2..maxTime) ( global_cardinality([ 
%  roster[r, h] | r in SHIFT_ROWs],
%  [oo],
%  [max_required_live_classes_at_any_timeblock - minNeeded[lp,h-1] - minNeeded[ls,h-1] - minNeeded[lc,h-1] 
%  - minNeeded[pp,h] - minNeeded[ps,h] - minNeeded[pc,h] - minNeeded[lp,h] - minNeeded[ls,h] - minNeeded[lc,h] +
%  continuedBlocks[ls,h] + continuedBlocks[lp,h] + continuedBlocks[lc,h]]
%  ));
 
 constraint forall(h in 1..1) ( global_cardinality([ 
 roster[r, h] | r in SHIFT_ROWs],
 [oo],
 [max_required_live_classes_at_any_timeblock - minNeeded[pp,h] - minNeeded[ps,h] 
 - minNeeded[pc,h] - minNeeded[lp,h] - minNeeded[ls,h] - minNeeded[lc,h] ] 
 ));


% constraint forall (r in SHIFT_ROWs, h in 1..maxTime-3) (roster_position[r,h] in SHIFT_INIT_BLOCK ->
% forall (i in h..h+3) ( 

% roster[r,h] in portugueseShifts <-> roster[r,i] in portugueseShifts 
% \/

%  roster[r,h] in spanishShifts <-> roster[r,i] in spanishShifts
% %\/
% %not ( roster[r,h] in spanishShifts) <-> not(roster[r,i] in spanishShifts)

% ));


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

constraint forall(r in SHIFT_ROWs) (regular([roster_position[r,h] | h in TIME], Q, S, transition_array, q0, STEPS));                                   

var int: obj = sum([ ( values[roster[r,h]]   ) | r in SHIFT_ROWs, h in TIME] ); 
                                   
solve satisfy;                                         
%solve maximize obj;
%output [show(roster) ++ show(roster_position)];
output [show(roster)]  ++ ["\n"] ++ [show(roster_position)];

%output [ show(SHIFT_ROWs[i]) ++ ";" ++ timeLabels[j] ++ ";" ++ blockName[fix(roster[i,j])] ++ "\n"  | i in SHIFT_ROWs, j in TIME ]  ;