# Load required packages
library(ncaahoopR)
library(dplyr)

#' Get all NCAA basketball teams.
#' 
#' @return A character vector of team names.
get_all_ncaa_teams <- function() {
  data(ids)
  return(ids$team)
}

#' Get box scores for a team's entire season.
#'
#' @param team_name The name of the team.
#' @param season The NCAA basketball season (e.g., "2024-25").
#' @return A dataframe of the team's box scores or NULL if none found.
get_team_season_box_scores <- function(team_name, season = NULL) {
  dir_name <- "ncaa_box_scores_2024_25"
  if (!dir.exists(dir_name)) {
    dir.create(dir_name)
  }

  cat(sprintf("\nProcessing team: %s for season %s\n", team_name, season))

  tryCatch({
    schedule <- get_schedule(team_name, season)

    if (nrow(schedule) == 0) {
      cat(sprintf("No games found for %s in season %s\n", team_name, season))
      return(NULL)
    }

    all_box_scores <- data.frame()

    for (i in 1:nrow(schedule)) {
      game_id <- schedule$game_id[i]
      opponent <- schedule$opponent[i]
      date <- schedule$date[i]
      location <- schedule$location[i]

      tryCatch({
        box_score <- get_boxscore(game_id)
        box_names <- names(box_score)

        team_box_idx <- which(grepl(team_name, box_names, ignore.case = TRUE))

        if (length(team_box_idx) == 0) {
          team_box_idx <- 1
          if (grepl(opponent, box_names[1], ignore.case = TRUE)) {
            team_box_idx <- 2
          }
        }

        team_box <- box_score[[team_box_idx]]
        opp_box_idx <- ifelse(team_box_idx == 1, 2, 1)
        opponent_box <- box_score[[opp_box_idx]]

        team_score <- sum(team_box$PTS, na.rm = TRUE)
        opponent_score <- sum(opponent_box$PTS, na.rm = TRUE)

        result <- ifelse(team_score > opponent_score, "W", 
                         ifelse(team_score < opponent_score, "L", "T"))

        team_box$game_id <- game_id
        team_box$date <- date
        team_box$opponent <- opponent
        team_box$location <- location
        team_box$team_score <- team_score
        team_box$opponent_score <- opponent_score
        team_box$result <- result

        if (nrow(all_box_scores) == 0) {
          all_box_scores <- team_box
        } else {
          common_cols <- intersect(names(all_box_scores), names(team_box))
          all_box_scores <- rbind(all_box_scores[, common_cols], 
                                  team_box[, common_cols])
        }

        cat(sprintf("  Game %d of %d: vs %s - Result: %s (%d-%d)\n", 
                    i, nrow(schedule), opponent, result,
                    team_score, opponent_score))

        Sys.sleep(1)
      }, error = function(e) {
        cat(sprintf("  Error processing game %d (ID: %s): %s\n", 
                    i, game_id, e$message))
      })
    }

    if (nrow(all_box_scores) > 0) {
      team_filename <- file.path(dir_name, paste0(gsub(" ", "_", team_name), "_2024_25.csv"))
      write.csv(all_box_scores, team_filename, row.names = FALSE)
      cat(sprintf("CSV file saved: %s\n", team_filename))
      return(all_box_scores)
    } else {
      cat(sprintf("No box scores data collected for %s\n", team_name))
      return(NULL)
    }
  }, error = function(e) {
    cat(sprintf("Error with team %s: %s\n", team_name, e$message))
    return(NULL)
  })
}

# -------- Main execution --------

season <- "2024-25"

# Get all teams
all_teams <- get_all_ncaa_teams()
cat(sprintf("Found %d NCAA basketball teams\n", length(all_teams)))

# Process each team
results <- list()
for (team in all_teams) {
  cat(sprintf("\n======= Processing %s =======\n", team))
  result <- get_team_season_box_scores(team, season)
  results[[team]] <- result
  Sys.sleep(2)
}

cat("\nProcessing complete! Box scores saved to 'ncaa_box_scores_2024_25' directory.\n")
