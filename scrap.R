library(rvest)
library(stringr)
library(dplyr)

# Paso 1: Descarga el HTML
url <- "https://www.worldcubeassociation.org/persons/2019WANY36"
page <- read_html(url)

# Paso 2: Extrae los scripts que contengan 'competitions' o 'marker_date'
scripts <- page %>% html_nodes("script") %>% html_text()
script_json <- scripts[grep("marker_date", scripts)[1]]  # toma el primero que contenga datos

# Paso 3: Extraer el contenido de JSON crudo que está en el script
# Suponiendo que haya un array: var COMPETITIONS = [ {...}, {...}, ... ];
json_raw <- str_extract(script_json, "\\[\\{.*\\}\\]")

# Paso 4: Si json_raw extrae bien, puedes parsearlo con jsonlite
library(jsonlite)
df_all <- fromJSON(json_raw, flatten = TRUE)

# Paso 5: Filtrar competencias de 2024, 2025 o con "avg" en el 'name'
df_filtered <- df_all %>%
  filter(
           str_detect(tolower(name), "Average"))

# Paso 6: Mostrar resultado
print(df_filtered)

write.csv(df_all, 'yiheng.csv')
