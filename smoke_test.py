dataset,column,korean_name,description,dtype,non_null_count,missing_count,missing_rate
matches_clean,match_id,경기 ID,시즌 안에서 유일한 경기 식별자,string,7650,0,0.0
matches_clean,season,시즌,분데스리가 시즌,string,7650,0,0.0
matches_clean,season_start_year,시즌 시작 연도,시즌이 시작된 연도,Int64,7650,0,0.0
matches_clean,match_sequence_in_season,시즌 경기 순번,날짜 기준 시즌 내 전체 경기 순번,int64,7650,0,0.0
matches_clean,match_date,경기일,경기 날짜,datetime64[ns],7650,0,0.0
matches_clean,home_team,홈팀,표준화된 홈팀 이름,string,7650,0,0.0
matches_clean,away_team,원정팀,표준화된 원정팀 이름,string,7650,0,0.0
matches_clean,home_goals,홈 득점,정규시간 홈팀 득점,Int64,7650,0,0.0
matches_clean,away_goals,원정 득점,정규시간 원정팀 득점,Int64,7650,0,0.0
matches_clean,result_code,결과 코드,"H=홈승, D=무승부, A=원정승",string,7650,0,0.0
matches_clean,result_label,결과 영문,영문 경기 결과,object,7650,0,0.0
matches_clean,result_label_ko,결과 한글,한글 경기 결과,object,7650,0,0.0
matches_clean,winner,승리팀,승리팀 또는 Draw,object,7650,0,0.0
matches_clean,home_points,홈 승점,경기에서 홈팀이 획득한 승점,Int64,7650,0,0.0
matches_clean,away_points,원정 승점,경기에서 원정팀이 획득한 승점,Int64,7650,0,0.0
matches_clean,goal_difference_home,홈 기준 득실차,홈 득점-원정 득점,Int64,7650,0,0.0
matches_clean,total_goals,총득점,양 팀 득점 합계,Int64,7650,0,0.0
matches_clean,over_2_5_goals,2.5골 초과,총득점 3골 이상 여부,boolean,7650,0,0.0
matches_clean,both_teams_scored,양 팀 득점,양 팀 모두 한 골 이상 득점했는지,boolean,7650,0,0.0
matches_clean,half_time_home_goals,Half Time Home Goals,정제·파생된 분석 변수,Int64,7649,1,0.0001
matches_clean,half_time_away_goals,Half Time Away Goals,정제·파생된 분석 변수,Int64,7649,1,0.0001
matches_clean,half_time_result_code,Half Time Result Code,정제·파생된 분석 변수,string,7649,1,0.0001
matches_clean,second_half_home_goals,Second Half Home Goals,정제·파생된 분석 변수,Int64,7649,1,0.0001
matches_clean,second_half_away_goals,Second Half Away Goals,정제·파생된 분석 변수,Int64,7649,1,0.0001
matches_clean,home_shots,Home Shots,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,away_shots,Away Shots,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,home_shots_on_target,Home Shots On Target,정제·파생된 분석 변수,Int64,6425,1225,0.1601
matches_clean,away_shots_on_target,Away Shots On Target,정제·파생된 분석 변수,Int64,6425,1225,0.1601
matches_clean,home_shot_accuracy,Home Shot Accuracy,정제·파생된 분석 변수,Float64,6424,1226,0.1603
matches_clean,away_shot_accuracy,Away Shot Accuracy,정제·파생된 분석 변수,Float64,6424,1226,0.1603
matches_clean,home_corners,Home Corners,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,away_corners,Away Corners,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,home_fouls,Home Fouls,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,away_fouls,Away Fouls,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,home_yellow_cards,Home Yellow Cards,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,away_yellow_cards,Away Yellow Cards,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,home_red_cards,Home Red Cards,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,away_red_cards,Away Red Cards,정제·파생된 분석 변수,Int64,7343,307,0.0401
matches_clean,attendance,Attendance,정제·파생된 분석 변수,Int64,306,7344,0.96
matches_clean,referee,Referee,정제·파생된 분석 변수,string,306,7344,0.96
matches_clean,match_stats_available,기본 경기지표 보유,슈팅과 코너 지표가 모두 있는지,bool,7650,0,0.0
matches_clean,detailed_shots_available,유효슈팅 보유,양 팀 유효슈팅 지표가 있는지,bool,7650,0,0.0
matches_clean,source_file,Source File,정제·파생된 분석 변수,string,7650,0,0.0
team_match_long,match_id,경기 ID,시즌 안에서 유일한 경기 식별자,string,15300,0,0.0
team_match_long,season,시즌,분데스리가 시즌,string,15300,0,0.0
team_match_long,season_start_year,시즌 시작 연도,시즌이 시작된 연도,Int64,15300,0,0.0
team_match_long,match_date,경기일,경기 날짜,datetime64[ns],15300,0,0.0
team_match_long,team,팀,표준화된 분석 대상 팀,string,15300,0,0.0
team_match_long,opponent,상대팀,표준화된 상대팀,string,15300,0,0.0
team_match_long,venue,경기장 구분,Home 또는 Away,object,15300,0,0.0
team_match_long,venue_ko,Venue Ko,정제·파생된 분석 변수,object,15300,0,0.0
team_match_long,goals_for,득점,해당 팀 득점,Int64,15300,0,0.0
team_match_long,goals_against,실점,해당 팀 실점,Int64,15300,0,0.0
team_match_long,goal_difference,득실차,득점-실점,Int64,15300,0,0.0
team_match_long,result,팀 기준 결과,"W=승, D=무, L=패",object,15300,0,0.0
team_match_long,points,승점,해당 팀 획득 승점,Int64,15300,0,0.0
team_match_long,shots_for,Shots For,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,shots_against,Shots Against,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,shots_on_target_for,Shots On Target For,정제·파생된 분석 변수,Int64,12850,2450,0.1601
team_match_long,shots_on_target_against,Shots On Target Against,정제·파생된 분석 변수,Int64,12850,2450,0.1601
team_match_long,corners_for,Corners For,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,corners_against,Corners Against,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,fouls_committed,Fouls Committed,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,fouls_drawn,Fouls Drawn,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,yellow_cards,Yellow Cards,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,red_cards,Red Cards,정제·파생된 분석 변수,Int64,14686,614,0.0401
team_match_long,clean_sheet,Clean Sheet,정제·파생된 분석 변수,boolean,15300,0,0.0
team_match_long,scored,Scored,정제·파생된 분석 변수,boolean,15300,0,0.0
team_match_long,match_stats_available,기본 경기지표 보유,슈팅과 코너 지표가 모두 있는지,bool,15300,0,0.0
team_match_long,team_match_number,Team Match Number,정제·파생된 분석 변수,int64,15300,0,0.0
team_match_long,rolling_points_last5,Rolling Points Last5,정제·파생된 분석 변수,Int64,15300,0,0.0
team_match_long,rolling_goal_diff_last5,Rolling Goal Diff Last5,정제·파생된 분석 변수,Int64,15300,0,0.0
team_season_summary,season,시즌,분데스리가 시즌,string,450,0,0.0
team_season_summary,season_start_year,시즌 시작 연도,시즌이 시작된 연도,Int64,450,0,0.0
team_season_summary,calculated_rank,계산 순위,"승점, 득실차, 다득점 순으로 계산한 순위",int64,450,0,0.0
team_season_summary,team,팀,표준화된 분석 대상 팀,string,450,0,0.0
team_season_summary,matches,Matches,정제·파생된 분석 변수,int64,450,0,0.0
team_season_summary,wins,Wins,정제·파생된 분석 변수,int64,450,0,0.0
team_season_summary,draws,Draws,정제·파생된 분석 변수,int64,450,0,0.0
team_season_summary,losses,Losses,정제·파생된 분석 변수,int64,450,0,0.0
team_season_summary,goals_for,득점,해당 팀 득점,Int64,450,0,0.0
team_season_summary,goals_against,실점,해당 팀 실점,Int64,450,0,0.0
team_season_summary,goal_difference,득실차,득점-실점,Int64,450,0,0.0
team_season_summary,points,승점,해당 팀 획득 승점,Int64,450,0,0.0
team_season_summary,points_per_match,Points Per Match,정제·파생된 분석 변수,Float64,450,0,0.0
team_season_summary,win_rate,Win Rate,정제·파생된 분석 변수,float64,450,0,0.0
team_season_summary,clean_sheets,Clean Sheets,정제·파생된 분석 변수,Int64,450,0,0.0
team_season_summary,clean_sheet_rate,Clean Sheet Rate,정제·파생된 분석 변수,Float64,450,0,0.0
team_season_summary,home_points,홈 승점,경기에서 홈팀이 획득한 승점,Int64,450,0,0.0
team_season_summary,away_points,원정 승점,경기에서 원정팀이 획득한 승점,Int64,450,0,0.0
team_season_summary,avg_shots_for,Avg Shots For,정제·파생된 분석 변수,Float64,432,18,0.04
team_season_summary,avg_shots_against,Avg Shots Against,정제·파생된 분석 변수,Float64,432,18,0.04
players_clean,player_club_id,선수-구단 ID,2025/26 선수와 구단 조합 식별자,object,507,0,0.0
players_clean,season,시즌,분데스리가 시즌,object,507,0,0.0
players_clean,player,선수,선수 이름,string,507,0,0.0
players_clean,nationality_code,국적 코드,3자리 국가 코드,string,507,0,0.0
players_clean,position_raw,원 포지션,복수 포지션을 포함한 원자료 포지션,string,507,0,0.0
players_clean,primary_position,주 포지션,원자료의 첫 번째 포지션,object,507,0,0.0
players_clean,club,소속팀,표준화된 소속팀,string,507,0,0.0
players_clean,age,Age,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,birth_year,Birth Year,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,matches,Matches,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,starts,Starts,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,minutes,출전시간,시즌 총 출전시간,Int64,507,0,0.0
players_clean,nineties,90분 환산,출전시간을 90분 단위로 환산,float64,507,0,0.0
players_clean,goals,Goals,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,assists,Assists,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,non_penalty_goals,Non Penalty Goals,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,penalties_scored,Penalties Scored,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,penalties_attempted,Penalties Attempted,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,yellow_cards,Yellow Cards,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,red_cards,Red Cards,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,shots,Shots,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,shots_on_target,Shots On Target,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,shot_on_target_pct,Shot On Target Pct,정제·파생된 분석 변수,float64,427,80,0.1578
players_clean,goals_per_shot,Goals Per Shot,정제·파생된 분석 변수,float64,427,80,0.1578
players_clean,goals_per_shot_on_target,Goals Per Shot On Target,정제·파생된 분석 변수,float64,363,144,0.284
players_clean,minutes_per_match,Minutes Per Match,정제·파생된 분석 변수,int64,507,0,0.0
players_clean,team_minutes_pct,Team Minutes Pct,정제·파생된 분석 변수,float64,507,0,0.0
players_clean,minutes_per_start,Minutes Per Start,정제·파생된 분석 변수,float64,440,67,0.1321
players_clean,completed_matches,Completed Matches,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,sub_appearances,Sub Appearances,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,unused_sub,Unused Sub,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,points_per_match,Points Per Match,정제·파생된 분석 변수,float64,507,0,0.0
players_clean,team_goals_while_on_pitch,Team Goals While On Pitch,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,team_goals_against_while_on_pitch,Team Goals Against While On Pitch,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,plus_minus,Plus Minus,정제·파생된 분석 변수,int64,507,0,0.0
players_clean,plus_minus_per90,Plus Minus Per90,정제·파생된 분석 변수,float64,507,0,0.0
players_clean,on_off_per90,On Off Per90,정제·파생된 분석 변수,float64,495,12,0.0237
players_clean,crosses,Crosses,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,tackles_won,Tackles Won,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,interceptions,Interceptions,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,fouls_committed,Fouls Committed,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,fouls_drawn,Fouls Drawn,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,offsides,Offsides,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,own_goals,Own Goals,정제·파생된 분석 변수,Int64,507,0,0.0
players_clean,goals_against_gk,Goals Against Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,goals_against_per90_gk,Goals Against Per90 Gk,정제·파생된 분석 변수,float64,30,477,0.9408
players_clean,shots_on_target_against_gk,Shots On Target Against Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,saves_gk,Saves Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,save_pct_gk,Save Pct Gk,정제·파생된 분석 변수,float64,30,477,0.9408
players_clean,wins_gk,Wins Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,draws_gk,Draws Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,losses_gk,Losses Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,clean_sheets_gk,Clean Sheets Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,clean_sheet_pct_gk,Clean Sheet Pct Gk,정제·파생된 분석 변수,float64,30,477,0.9408
players_clean,penalties_faced_gk,Penalties Faced Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,penalties_allowed_gk,Penalties Allowed Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,penalties_saved_gk,Penalties Saved Gk,정제·파생된 분석 변수,Int64,30,477,0.9408
players_clean,position_group,포지션 그룹,"Goalkeeper, Defender, Midfielder, Forward",object,507,0,0.0
players_clean,position_group_ko,Position Group Ko,정제·파생된 분석 변수,object,507,0,0.0
players_clean,age_group,Age Group,정제·파생된 분석 변수,string,507,0,0.0
players_clean,goal_contributions,공격포인트,득점+도움,Int64,507,0,0.0
players_clean,goals_per90,Goals Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,assists_per90,Assists Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,goal_contributions_per90,Goal Contributions Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,non_penalty_goals_per90,Non Penalty Goals Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,shots_per90,Shots Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,shots_on_target_per90,Shots On Target Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,crosses_per90,Crosses Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,tackles_won_per90,Tackles Won Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,interceptions_per90,Interceptions Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,fouls_committed_per90,Fouls Committed Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,fouls_drawn_per90,Fouls Drawn Per90,정제·파생된 분석 변수,Float64,507,0,0.0
players_clean,shot_conversion_pct,Shot Conversion Pct,정제·파생된 분석 변수,Float64,427,80,0.1578
players_clean,ranking_eligible_450min,450분 랭킹 대상,450분 이상 출전 여부,boolean,507,0,0.0
players_clean,ranking_eligible_900min,900분 랭킹 대상,900분 이상 출전 여부,boolean,507,0,0.0
players_clean,sample_size_group,표본 구간,출전시간에 따른 소·중·정규 표본 구간,object,507,0,0.0
players_clean,is_goalkeeper,Is Goalkeeper,정제·파생된 분석 변수,bool,507,0,0.0
players_clean,multi_club_player,복수 구단 선수,시즌 중 두 구단 기록이 존재하는 선수 여부,bool,507,0,0.0
players_clean,stat_coverage_pct,지표 보유율,해당 행에서 값이 존재하는 컬럼 비율,float64,507,0,0.0
