# Template R Operator

The `Template R Operator` is a template repository for the creation of R operators in Tercen. An overview of steps for developing an operator are:

1. Create a new GitHub repository using this template
2. Clone the newly created repository to your development environment (we recommend using VS Code)
3. Describe your operator specifications in the README file
4. Develop the operator (with or without assistance from the Tercen Agents)
5. Initialise or update the R packages environment using `renv`.
6. Push your changes and install the operator in Tercen

Detailed information can be found in the [Tercen developer's guide](https://tercen.github.io/developers_guide/).

Below is the operator README standard structure.

### Description

The `Template R operator` is a template repository for the creation of R operators in Tercen.

### Usage

Input|.
---|---
`x-axis`        | type, description 
`y-axis`        | type, description 
`row`           | type, description 
`column`        | type, description 
`colors`        | type, description 
`labels`        | type, description 

Settings|.
---|---
`input_var`        | parameter description

Output|.
---|---
`output_var`        | output relation
`Operator view`        | view of the Shiny application

### Details

Details on the computation.
