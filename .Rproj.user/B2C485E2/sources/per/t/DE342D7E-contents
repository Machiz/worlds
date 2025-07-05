library(chromote)
library(rvest)
library(dplyr)

# Competidor WCA
wca_id <- "2016KOLA02"
url <- paste0("https://www.worldcubeassociation.org/persons/", wca_id)

# Iniciar navegador virtual con Chromote
b <- ChromoteSession$new()

# Ir a la página
b$Page$navigate(url)
b$Page$loadEventFired()

# Esperar a que exista la tabla en el DOM
b$Runtime$evaluate(
  expression = "new Promise(resolve => {
    const check = () => {
      if (document.querySelector('#results')) {
        resolve(true);
      } else {
        setTimeout(check, 500);
      }
    };
    check();
  })",
  awaitPromise = TRUE
)

# Obtener HTML renderizado
html_id <- b$DOM$getDocument()$root$nodeId
full_html <- b$DOM$getOuterHTML(nodeId = html_id)$outerHTML
page <- read_html(full_html)

# Ahora sí: extraer la tabla
results_table <- page %>%
  html_element("table#results") %>%
  html_table()

# Guardar CSV
output_file <- paste0("resultados_", wca_id, ".csv")
write.csv(results_table, output_file, row.names = FALSE)

# Confirmación
cat("✅ Resultados guardados en '", output_file, "' con ", nrow(results_table), " filas.\n", sep = "")
