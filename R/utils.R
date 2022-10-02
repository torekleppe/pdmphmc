.from.jsonFile <- function(fileName){
  return( jsonlite::fromJSON(readLines(con=fileName)))
}
