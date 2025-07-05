library(jsonlite)

# ID del competidor (puedes reemplazarlo por cualquier otro)
wca_id <- "2016KOLA02"

# URL de la API de la WCA
url <- paste0("https://www.worldcubeassociation.org/api/v0/persons/", wca_id)

# Descargar y parsear JSON
data <- fromJSON(url, flatten = TRUE)

# Extraer tabla completa de competencias con todos los resultados
competition_results <- data$competition_results

# Guardar como CSV con todas las columnas
write.csv(competition_results, paste0("competencias_completas_", wca_id, ".csv"), row.names = FALSE)

# Confirmar
cat("✅ Se guardó el archivo 'competencias_completas_", wca_id, ".csv' con ", nrow(competition_results), " filas y ", ncol(competition_results), " columnas.\n", sep = "")
