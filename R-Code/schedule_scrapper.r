# NCAAB Schedule Scraper
# This program fetches the NCAA basketball schedule for a date range
# and saves it to a CSV file with team names and game IDs

library(ncaahoopR)
library(dplyr)
library(purrr)
library(lubridate)

#' Retrieve the NCAA basketball schedule for a specific date.
#'
#' @param date A character string in "YYYY-MM-DD" format representing the date.
#' @return A dataframe containing the schedule for the specified date, or NULL if an error occurs.
get_date_schedule <- function(date) {
  tryCatch({
    schedule <- get_master_schedule(date)
    schedule$date <- date  # Add date column for reference
    return(schedule)
  }, error = function(e) {
    message(paste("Error retrieving schedule for", date, ":", e$message))
    return(NULL)
  })
}

#' Retrieve NCAA basketball schedules for a range of dates.
#'
#' @param start_date The start date in "YYYY-MM-DD" format.
#' @param end_date The end date in "YYYY-MM-DD" format.
#' @return A dataframe containing schedules for all dates in the range.
get_schedule_range <- function(start_date, end_date) {
  dates <- seq(as.Date(start_date), as.Date(end_date), by = "days")
  total_dates <- length(dates)
  cat("Fetching schedule for", total_dates, "dates\n")

  all_schedules <- map_df(seq_along(dates), function(i) {
    date <- dates[i]
    date_str <- format(date, "%Y-%m-%d")
    cat(sprintf("Processing date %d/%d: %s\n", i, total_dates, date_str))
    schedule <- get_date_schedule(date_str)
    Sys.sleep(0.5)  # Delay to prevent rate-limiting
    return(schedule)
  })

  return(all_schedules)
}

#' Main function to execute the NCAAB schedule scraping process.
#'
#' Fetches schedule data for the 2024–25 season and writes it to a CSV file.
main <- function() {
  start_date <- "2024-11-01"  # Season start
  end_date <- "2025-04-10"    # Season end
  cat("Starting NCAAB schedule scraping process...\n")
  cat("Date range:", start_date, "to", end_date, "\n")

  all_schedules <- get_schedule_range(start_date, end_date)

  if (is.null(all_schedules) || nrow(all_schedules) == 0) {
    stop("No schedule data was retrieved.")
  }

  schedule_data <- all_schedules %>%
    select(date, game_id, home, away) %>%
    mutate(date = as.character(date))

  output_file <- "schedule.csv"
  write.csv(schedule_data, output_file, row.names = FALSE)

  cat("\nSchedule data has been written to", output_file, "\n")
  cat("Total games in schedule:", nrow(schedule_data), "\n")
}

# Run the script
main()
