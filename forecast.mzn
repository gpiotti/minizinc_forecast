include "global_cardinality_low_up.mzn";
include "global_cardinality.mzn";
include "at_most.mzn";

array[TIME] of string: timeLabels;
array[1..card(BLOCK)] of string: blockName;

% blocks
enum BLOCK = {private_spanish, private_port, private_classic,
              live_spanish, live_port, live_classic,
              block_cont, idle, off};
        

set of BLOCK: STD_BLOCK = { private_spanish, private_port, private_classic,
              live_spanish, live_port, live_classic };
              
set of BLOCK: LIVE_BLOCK = { live_spanish, live_port, live_classic };
              
set of BLOCK: PRIVATE_BLOCK = { private_spanish, private_port, private_classic };  

set of BLOCK: SPANISH_BLOCK = { private_spanish, live_spanish };  

set of BLOCK: PORT_BLOCK = {  private_port, live_port};  

set of BLOCK: ONSHIFT_BLOCK = { block_cont, idle } ;
                          
set of int: BLOCKs = 1..card(BLOCK);

%shifts
enum SHIFT = {tbp, tbss, classic};


int: maxTime;
int: max_required_at_any_timeblock= 10;
set of int: TIME = 1..maxTime;  


array[STD_BLOCK, TIME] of int: minNeeded;

set of int: SHIFT_ROWs = 1..max_required_at_any_timeblock * 2;


%variables
array[SHIFT_ROWs, TIME] of var BLOCK: roster;

%array[SHIFT_ROWs, TIME] of var SHIFT: shifts_schedule;

%global cardinality low up
% constraint forall(h in TIME) ( global_cardinality_low_up([roster[r, h] | r in SHIFT_ROWs],
%  [private_spanish, private_port, private_classic,
%   live_spanish, live_port, live_classic, off],
%  [0, 0, 0, 0, 0, 0 , 0 ],
%  [ minNeeded[private_spanish,h], minNeeded[private_port,h],  minNeeded[private_classic,h],
%    minNeeded[live_spanish,h], minNeeded[live_port,h],  minNeeded[live_classic,h],
%    max_required_at_any_timeblock * 2] ) );

constraint forall(h in TIME) ( global_cardinality([roster[r, h] | r in SHIFT_ROWs],
 [private_spanish, private_port, private_classic,
  live_spanish, live_port, live_classic],
 [ minNeeded[private_spanish,h], minNeeded[private_port,h],  minNeeded[private_classic,h],
   minNeeded[live_spanish,h], minNeeded[live_port,h],  minNeeded[live_classic,h]] ) );

%constraint global_cardinality([roster[r, 1] | r in SHIFT_ROWs], [idle, block_cont], [0, 0]);


constraint global_cardinality([roster[r, 1] | r in SHIFT_ROWs], [block_cont], [0]);
constraint global_cardinality([roster[r, 1] | r in SHIFT_ROWs], [idle], [0]);


   
%live class span 2 blocks
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-1) (
                                               roster[r,h] in LIVE_BLOCK 
                                               <-> roster[r,h+1] = block_cont);
% max shift duration = 4 blocks
%constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-3) (
%                                               roster[r,h] in STD_BLOCK 
%                                               -> (roster[r,h+1] != off /\ roster[r,h+2] != off /\ roster[r,h+3] != off));
                                               
% shifts can not have different language blocks
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-3) (
                                               roster[r,h] in SPANISH_BLOCK 
                                               -> not (roster[r,h+1]  in PORT_BLOCK \/
                                                       roster[r,h+2]  in PORT_BLOCK \/
                                                       roster[r,h+3]  in PORT_BLOCK ));
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-3) (
                                               roster[r,h] in PORT_BLOCK 
                                               -> not (roster[r,h+1]  in SPANISH_BLOCK \/
                                                       roster[r,h+2]  in SPANISH_BLOCK \/
                                                       roster[r,h+3]  in SPANISH_BLOCK ));                                               

                                               
% shift can not start with idle nor block_cont
%constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-1) ( roster[r,h] == off  -> (roster[r,h+1] != idle /\ roster[r,h+1] != block_cont));
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-1) ( roster[r,h] == off  -> not (roster[r,h+1] in ONSHIFT_BLOCK));  
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-4) ( roster[r,h] in STD_BLOCK -> not (roster[r, h+4] in {idle}));

%shift must finish only with a private class or a block_cont
constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-3) ( roster[r,h] in STD_BLOCK  -> roster[r,h+3] in (PRIVATE_BLOCK union { block_cont, idle }) );

%start_times
%constraint forall(r in SHIFT_ROWs,  h in 1..length(TIME)-1) ( roster[r,h] == off  -> shifts_schedule[r,h+1] = tbp);  
%constraint forall(r in SHIFT_ROWs) ( roster[r,1] in STD_BLOCK -> start_times[r,1] == 1);   

%constraint forall(r in SHIFT_ROWs,  h in TIME) ( roster[r,h] != off  -> start_times[r,h] = 0 ); 
                                         
                                          
solve satisfy;
%output [show(shifts_schedule)];
output [ show(SHIFT_ROWs[i]) ++ ";" ++ timeLabels[j] ++ ";" ++ blockName[fix(roster[i,j])] ++ "\n"  | i in SHIFT_ROWs, j in TIME ]  ;