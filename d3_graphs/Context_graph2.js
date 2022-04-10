
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// var x_input,y_input,
//     svg_input_Xaxis,
//     svg_input_Yaxis,
//     values,
//     data,
//     data_output,
//     svg_heatmap,
//     legend,
//     svg_input,
//     svg_output,
//     height,
//     width,
//     margin

// var parseDate = d3.timeParse("%d/%m/%Y %H:%M");
var input_graphs = []
// var chart_type = []
var input_scales = []
var input_axes = []
var input_lines = []

var heatmap_graphs = []
var heatmap_graphs_2 = []
var heatmap_scales = []
var heatmap_axes = []

var dates_scale = []
var dates_axes = []
var dates_min_axes = []
var dates_noon_axes = []

var output_graphs = []
var output_scales = []
var output_axes = []
var output_lines = []
var output_line_true = []

var values = [] 
var output_values = []
var charts = []
var axis_date_groups = []
var heatmap_groups = []
var input_groups = []


var legend_graphs = []
var legend_scales = []
var legend_axes = []


// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




function context_graph(context_data, output_data, name) {

  var scales = new Array()
  var axes = new Array()

  var scales_heat = new Array()
  var axes_heat = new Array()

  var scales_out = new Array()
  var axes_out = new Array()

  var scale_dates = new Array()
  var axes_dates = new Array()
  var axes_dates_min = new Array()
  var axes_dates_noon = new Array()

  charts.push(name)

// function instaniate_charts() {
  // set the dimensions and margins of the graph
  var margin = {top: 0, right: 0, bottom: 0, left: 25},
  // width = 1500 
  height = 275

  var width = parseInt(d3.select('.slideshow-container').style('width'), 10) + 150


  // append the svg object to the body of the page
 heatmap_graphs['heatmap_' + name] = d3.select("#my_dataviz" + name + "2")
  .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr('cursor', 'crosshair')
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

  // heatmap legend
  legend_graphs['legend_' + name] = d3.select('#legend' + name)
  .append("svg")
    .attr("width", 80)
    .attr("height", height)
    .attr('cursor', 'crosshair')
  .append("g")
    .attr("transform",
          "translate(" + (margin.left+20) + "," + 20 + ")");



  // input line graph
  input_graphs['svg_input_' + name] = d3.select("#my_dataviz" + name + "3")
  .append("svg")
    .attr("width", width)
    .attr("height", height - 190) 
    .attr('cursor', 'crosshair')  
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


    // Add output line graph
  output_graphs['svg_output_' + name] = d3.select("#my_dataviz" + name + "4")
    .append("svg")
      .attr("class", 'output_svg_back' + name)
      .attr("width", width)
      .attr("height", height)
      .attr('cursor', 'crosshair')
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    // Add title 
    // d3.select("#my_dataviz3")
    //     .append("text")
    //     .attr("x", (width / 2))             
    //     .attr("y", 0 - (margin.top + 0))
    //     .attr("text-anchor", "middle")  
    //     .style("font-size", "16px") 
    //     .style("text-decoration", "underline")  
    //     .text("Value vs Date Graph");



  // Add chart title
  // input_graphs['svg_input_' + name].append("text")
  //         .attr("x", (width/2)-margin.right - margin.left)             
  //         .attr("y", 10)
  //         .style("font-size", "16px") 
  //         .style("text-decoration", "underline")  
  //         .text(name);


  // update current width of graphs
  // width = parseInt(d3.select('#test').style('width'), 10) + 60
  height = 250

  dimensions = width_check(width, height)
  main_graph_width = dimensions[0]
  buffer = dimensions[1]
  output_graph_width = dimensions[2]


  var adj_width = width - buffer

  output_graphs['svg_output_' + name]
  .attr("transform",
      "translate(" + adj_width + "," + 0 + ")");

// INPUT LINE SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////

scales['xscale'] = d3.scaleLinear()
  .range([0, main_graph_width])
  // .domain([0, 100])

scales['yscale'] = d3.scaleLinear()
  // .domain([0, 0.5])
  .range([ height/4, 0 ]);

axes['xAxis'] = d3.axisBottom().scale(scales['xscale']).tickSizeOuter(0).tickValues([]);
axes['yAxis'] = d3.axisLeft().scale(scales['yscale']).tickSizeOuter(0).tickValues([]);

input_lines[name] = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return scales['xscale'](d.input_time_ref) })
                .y(function(d) { return scales['yscale'](d.input_solar) })


// HEATMAP SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////


scales_heat['xscale'] = d3.scaleBand()
  .range([0, main_graph_width])

scales_heat['yscale'] = d3.scaleBand()
  .range([height, 0])

axes_heat['xAxis'] = d3.axisBottom().scale(scales_heat['xscale']).tickSizeOuter(0).tickValues([]);
axes_heat['yAxis'] = d3.axisLeft().scale(scales_heat['yscale']).tickSizeOuter(0).tickValues([]);


scale_dates['xscale'] = d3.scaleTime()
  .range([ 0, main_graph_width])

scale_dates['yscale'] = d3.scaleTime()
  .range([ height, 0 ])

axes_dates['xAxis'] = d3.axisBottom().scale(scale_dates['xscale']).tickFormat(d3.timeFormat("")).ticks(d3.timeHour.every(24)).tickSize(16);


// axis for minor tick marks 
// x_min_ticks = d3.scaleTime()
//   .range([0, main_graph_width])

axes_dates_min['xAxis'] = d3.axisBottom().scale(scale_dates['xscale']).ticks(96).tickFormat('').tickSize(3)
axes_dates_min['yAxis']  = d3.axisLeft().scale(scale_dates['yscale']).ticks(12).tickFormat('').tickSize(3)


if (width < 550) {  
  axes_dates_noon['xAxis'] = d3.axisBottom().scale(scale_dates['xscale']).tickFormat(d3.timeFormat("%d/%m")).ticks(d3.timeHour.every(12)).tickSize(8).tickPadding(8);
  axes_dates['yAxis'] = d3.axisLeft().scale(scale_dates['yscale']).tickFormat(d3.timeFormat("%d/%m")).ticks(d3.timeHour.every(12)).tickSize(8); // Noon y-tick
} else {
  axes_dates_noon['xAxis'] = d3.axisBottom().scale(scale_dates['xscale']).tickFormat(d3.timeFormat("%a %d")).ticks(d3.timeHour.every(12)).tickSize(8).tickPadding(8);
  axes_dates['yAxis'] = d3.axisLeft().scale(scale_dates['yscale']).tickFormat(d3.timeFormat("%a %d")).ticks(d3.timeHour.every(12)).tickSize(8); // Noon y-tick
}


// get width of legend relative to screen size
// var legend_vars = legend_size(main_graph_width)
// var legend_width = legend_vars[0]
// var legend_start =  legend_vars[1]

// setup for legend
legend_scales['legendScale' + name] = d3.scaleLinear()
   .range([height, 0])

//Set up X axis
legend_axes['legendAxis' + name] = d3.axisBottom().scale(legend_scales['legendScale' + name])


// OUTPUT SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////

scales_out['xscale'] = d3.scaleLinear()
  .range([ 0, output_graph_width ]);
  // .domain([0, 100])

scales_out['yscale'] = d3.scaleLinear()
  // .domain([0, 0.5])
  .range([ height, 0 ]);

axes_out['xAxis'] = d3.axisBottom().scale(scales_out['xscale']).tickSizeOuter(0).tickValues([]);
axes_out['yAxis'] = d3.axisLeft().scale(scales_out['yscale']).tickSizeOuter(0).tickValues([]);

output_lines[name] = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return scales_out['xscale'](d.prediction) })
                .y(function(d) { return scales_out['yscale'](d.output_time_ref) })


output_line_true[name] = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return scales_out['xscale'](d.y_true) })
                .y(function(d) { return scales_out['yscale'](d.output_time_ref) })


// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Get the data
d3.csv(context_data, 

    function(d){
    return { group_index: d.group_index = +d.group_index,
             variable_index: d.variable_index = +d.variable_index,
             group: d3.timeParse("%d/%m/%Y %H:%M")(d.group), 
             variable: d3.timeParse("%d/%m/%Y %H:%M")(d.variable),
             value_scaled: d.value_scaled = +d.value_scaled,
             value: d.value = +d.value,
             input_time_ref: d.input_time_ref = +d.input_time_ref,
             input_time: d3.timeParse("%d/%m/%Y %H:%M")(d.input_time),
             input_solar: d.input_solar = +d.input_solar,
             input_cloud_cover: d.input_cloud_cover = +d.input_cloud_cover,
             output_time_ref: d.output_time_ref = +d.output_time_ref,
             output_time: d3.timeParse("%d/%m/%Y %H:%M")(d.output_time),
             prediction: d.prediction = +d.prediction,
             y_true: d.y_true = +d.y_true,
    }
  },


  function(data) {


    values[name] =  data 
    // var data_output = data_out

    // output_values[name] = data_out;

    // get min and max of the dataset
  var min = d3.min(values[name], function(d){return d.value_scaled;})
  var max = d3.max(values[name], function(d){return d.value_scaled;})

    var defs = heatmap_graphs['heatmap_' + name].append("defs");
    var color = ["#2C7BB6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c", "#f9d057","#f29e2e","#e76818","#d7191c"]
    var colorScale = d3.scaleLinear().range(color);

    // List of groups = header of the csv files
    var keys = values[name].columns.slice(1);

// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HEATMAP GRAPH

    scales_heat['xscale'].domain(d3.map(values[name], function(d){return d.group;}).keys())

    heatmap_groups['svg_heat_Xaxis' + name] = heatmap_graphs['heatmap_' + name].append("g")
      .attr("class", "x axis" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(axes_heat['xAxis'])
      .select(".domain").remove()

    scales_heat['yscale'].domain(d3.map(values[name], function(d){return d.variable;}).keys())
    var svg_heat_Yaxis = heatmap_graphs['heatmap_' + name].append("g")
      .attr("class", "y axis" + name)
      .call(axes_heat['yAxis'])



heatmap_graphs_2['heatmap' + name] = heatmap_graphs['heatmap_' + name].selectAll()
    .data(values[name], function(d) {return d.group+':'+d.variable;})
    .enter()
    .append("rect")
      .attr("x", function(d) { return scales_heat['xscale'](d.group)})
      .attr("y", function(d) { return scales_heat['yscale'](d.variable )})
      .attr("width", scales_heat['xscale'].bandwidth()+0.7)
      .attr("height", scales_heat['yscale'].bandwidth()+0.7)
      // .style("fill", function(d) { return colorScaleRainbow(colorInterpolateRainbow(d.value)); } )
      .style("opacity", 1.0)
      .style("stroke-width", 0)
      .style("stroke", "none")


    // Build color scale
    var myColor = d3.scaleSequential()
      .domain(d3.extent(values[name], function(d){return d.value_scaled;}))
      .interpolator(color);

      //scale the colors
    colorScale.domain(d3.extent(values[name],function(d){
      return d.value_scaled;
    }));


    scale_dates['xscale'].domain(d3.extent(values[name], function(d){return d.group;}))


    axis_date_groups['svg_heat_Xdates' + name]  = heatmap_graphs['heatmap_' + name].append("g")
      .attr("class", "x axis_dates" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(axes_dates['xAxis'])

    scale_dates['yscale'].domain(d3.extent(values[name], function(d){return d.variable;}))
    
    axis_date_groups['svg_heat_Ydates' + name] = heatmap_graphs['heatmap_' + name].append("g")
      .style("text-transform", "uppercase")
      .style("font-size", "10px")
      .call(axes_dates['yAxis'])
    .selectAll('text')
      .attr("transform", "translate(-20, -20) rotate(-90)")


    var svg_heat_Ydates_min = heatmap_graphs['heatmap_' + name].append("g")
      .call(axes_dates_min['yAxis'])
      .select(".domain").remove();


    axis_date_groups['svg_heat_noon_ticks' + name] = heatmap_graphs['heatmap_' + name].append("g")
      .attr("class", "x axis_dates_noon" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(axes_dates_noon['xAxis'])
      .style("text-transform", "uppercase")
      .style("font-size", "10px")
      .select(".domain").remove();


    axis_date_groups['svg_heat_min_ticks' + name]  = heatmap_graphs['heatmap_' + name].append("g")
      .attr("class", "x axis_date_min" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(axes_dates_min['xAxis'])
      // .select(".domain").remove();

    var ticks = d3.select("#my_dataviz" + name + "2").selectAll(".tick text");
    ticks.each(function(_,i){
        if(i%2== 1) d3.select(this).remove();
    });             
    
    // if (width < 550) {  
    // var ticks = d3.select("#my_dataviz" + name + "2").selectAll(".tick text");
    // ticks.each(function(_,i){
    //     if(i%2== 0) d3.select(this).remove();
    // });             
    // }





// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// INPUT GRAPH

    scales['xscale'].domain([0, d3.max(values[name], function(d) { return +d.input_time_ref; })])
    scales['yscale'].domain([0, d3.max(values[name], function(d) { return +d.input_solar; })])

    input_groups['svg_input_Xaxis' + name] = input_graphs['svg_input_' + name].append("g")
      .attr("class", "x axis" + name)
      .attr("transform", "translate(0," + height/4 + ")")
      .call(axes['xAxis']);

      // .select(".domain").remove();

    var svg_input_Yaxis = input_graphs['svg_input_' + name].append("g")
      .attr("class", "y axis" + name)
      .call(axes['yAxis'])
      .select(".domain").remove();

    // Add the line path.
    var path = input_graphs['svg_input_' + name].append("path")
      .datum(values[name])
      .attr("class", "line" + name)
      .attr("fill", "steelblue")
      .attr("fill-opacity", 0.2)
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1)
      .attr("d", input_lines[name](values[name]));

// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OUTPUT GRAPH

      // // Add x axis
      // var x2 = d3.scaleLinear()
      //   .domain([0, d3.max(data_out, function(d2) { return +d2.prediction; })])
      //   .range([ 0, height/2.75 ]);
      // svg_output.append("g")
      //   .attr("transform", "translate(" + 0 + "," + height + ")")
      //   .call(d3.axisBottom(x2).tickSizeOuter(0).tickSizeInner(0).tickValues([]))
      //   .select(".domain").remove();

      // // Add Y axis
      // var y2 = d3.scaleLinear()
      //   .domain([0, d3.max(data_out, function(d2) { return +d2.output_time; })])
      //   .range([ height, 0 ]);
      // svg_output.append("g")
      //   .call(d3.axisLeft(y2).tickSizeOuter(0).tickSizeInner(0).tickValues([]))
      //   // .select(".domain").remove();


      // // Add the line
      // path3 = svg_output.append("path") 
      //   .datum(data_out)
      //   .attr("fill", "steelblue")
      //   .attr("fill-opacity", 0.2)
      //   .attr("stroke", "steelblue")
      //   .attr("stroke-width", 1)
      //   .attr("d", d3.line()
      //     .curve(d3.curveBasis)
      //     .x(function(d2) { return x2(d2.prediction) })
      //     .y(function(d2) { return y2(d2.output_time) })
      //     )


    scales_out['xscale'].domain([0, d3.max(values[name], function(d) { return +d.prediction; })])
    scales_out['yscale'].domain([0, d3.max(values[name], function(d) { return +d.output_time_ref; })])

    var svg_output_Xaxis = output_graphs['svg_output_' + name].append("g")
      .attr("class", "x axis_out" + name)
      .attr("transform", "translate(" + 0 + "," + height + ")")
      .call(axes_out['xAxis'])
      .select(".domain").remove();

    var svg_output_Yaxis = output_graphs['svg_output_' + name] .append("g")
      .attr("class", "y axis_out" + name)
      .call(axes_out['yAxis'])
      // .select(".domain").remove();


    var path_out_true = output_graphs['svg_output_' + name].append("path")
      .datum(values[name])
      .attr("class", "line_output_true" + name)
      .attr("fill", "orange")
      .attr("fill-opacity", 0.05)
      .attr("stroke", "orange")
      .attr("stroke-width", 1.25)
      .attr("d", output_line_true[name](values[name]));


    // Add the line path.
    var path_out = output_graphs['svg_output_' + name].append("path")
      .datum(values[name])
      .attr("class", "line_output" + name)
      .attr("fill", "steelblue")
      .attr("fill-opacity", 0.2)
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1)
      .attr("d", output_lines[name](values[name]));






///////////////////////////////////////////////////////////////////////////
//////////// Get continuous color scale for the Rainbow ///////////////////
///////////////////////////////////////////////////////////////////////////

// var coloursRainbow = ["#fde725", "#dde318", "#bade28", "#95d840", "#75d054", "#56c667", "#3dbc74", "#29af7f", "#20a386", "#1f968b", "#238a8d", "#287d8e", "#2d718e", "#33638d", "#39558c", "#404688", "#453781", "#482576", "#481467", "#440154"];
// var coloursRainbow = ["#d7e1ee", "#cbd6e4", "#bfcbdb", "#b3bfd1", "#a4a2a8", "#df8879", "#c86558", "#b04238", "#991f17"];
var coloursRainbow = ["#2C7BB6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c", "#f9d057","#f29e2e","#e76818","#d7191c"]
// coloursRainbow.reverse()
var colourRangeRainbow = d3.range(0, 1, 1 / (coloursRainbow.length - 1));
colourRangeRainbow.push(1);
       
//Create color gradient
var colorScaleRainbow = d3.scaleLinear()
  .domain(colourRangeRainbow)
  .range(coloursRainbow)
  .interpolate(d3.interpolateHcl);

//Needed to map the values of the dataset to the color scale
var colorInterpolateRainbow = d3.scaleLinear()
  .domain(d3.extent(values[name], function(d){return d.value_scaled;}))
  .range([0,1]);

// // get min and max of the dataset
// var min = d3.min(values[name], function(d){return d.value;})
// var max = d3.max(values[name], function(d){return d.value;})

console.log(min)
console.log(max)
console.log(colourRangeRainbow)


var legendWidth = width - 200
    var legendHeight = 7.5

    //  make the heatmap legend
    // https://www.visualcinnamon.com/2016/05/smooth-color-legend-d3-svg-gradient/
    var defs = legend_graphs['legend_' + name].append("defs");

    //Append a linearGradient element to the defs and give it a unique id
    var linearGradient = defs.append("linearGradient")
        .attr("id", "linear-gradient" + name);


    //Append multiple color stops by using D3's data/enter step
    linearGradient.selectAll("stop")
        .data([
            {offset: "0%", color: "#2c7bb6"},
            {offset: "12.5%", color: "#00a6ca"},
            {offset: "25%", color: "#00ccbc"},
            {offset: "37.5%", color: "#90eb9d"},
            {offset: "50%", color: "#ffff8c"},
            {offset: "62.5%", color: "#f9d057"},
            {offset: "75%", color: "#f29e2e"},
            {offset: "87.5%", color: "#e76818"},
            {offset: "100%", color: "#d7191c"}
          ])
        .enter().append("stop")
        .attr("offset", function(d) { return d.offset; })
        .attr("stop-color", function(d) { return d.color; });


    //A color scale
    var colorScale = d3.scaleLinear()
        .range(["#2c7bb6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c",
                "#f9d057","#f29e2e","#e76818","#d7191c"]);

    //Append multiple color stops by using D3's data/enter step
    linearGradient.selectAll("stop")
        .data( colorScale.range())
        .enter().append("stop")
        .attr("offset", function(d,i) { return i/(colorScale.range().length-1); })
        .attr("stop-color", function(d) { return d; });



    //Draw the rectangle and fill with gradient
    legend_graphs['legend_' + name].append("rect")
        .attr("width", height)
        .attr("height", legendHeight)
        .attr("transform", "translate(" + "10" + ","+ height+") rotate(-90)")
        .style("fill", "url(#linear-gradient" + name +  ")");

    //Append title
    // legend.append("text")
    //   .attr("class", "legendTitle")
    //   .attr("x", 100)
    //   .attr("y", -1)
    //   .style("text-anchor", "middle")
    //   .style("fill", "#4F4F4F")
    //   .text("Context");

    leg_min = d3.min(values[name], function(d){return d.value;})
    leg_max = d3.max(values[name], function(d){return d.value;}) 

    console.log(leg_max)

    var colourRangeRainbow_leg = d3.range(leg_min, leg_max, leg_max / (coloursRainbow.length - 1));
    colourRangeRainbow_leg.push(leg_max);

    console.log(colourRangeRainbow_leg)



    legend_scales['legendScale' + name].domain([leg_min, leg_max])


    var legend_axis_ref = legend_graphs['legend_' + name].append("g")
      .attr("class", "x axis_legend")
      .attr("transform", "translate(" + (-legendWidth/legendWidth+1) + "," + (legendHeight) + ")")
      .call(legend_axes['legendAxis' + name].ticks(0).tickSize(0))
      .select(".domain").remove();

    // // manually append min and max to legend
    // legend_graphs['legend_' + name]
    //   .append("text")
    //   .attr("class", "legend_max_txt" + name)
    //   .attr("x", (main_graph_width - legend_start - 20))
    //   .attr("y", 18)
    //   .attr("font-family", "arial")
    //   .attr("font-size", "10.25")
    //   // .style("text-anchor", "end ")
    //   .text(d3.format(".4f")(max));

    // legend_graphs['legend_' + name]
    //   .append("text")
    //   .attr("class", "legend_min_txt" + name)
    //   .attr("x", legend_start - 10)
    //   .attr("y", 18)
    //   .attr("font-family", "arial")
    //   .attr("font-size", "10.25")
    //   // .style("text-anchor", "end ")
    //   .text(d3.format(".4f")(min));

    // legend_graphs['legend_' + name]
    //   .append("text")
    //   .attr("class", "legend_mid_txt" + name)
    //   .attr("x",(main_graph_width/2)-10)
    //   .attr("y", 18)
    //   .attr("font-family", "arial")
    //   .attr("font-size", "10.25")
    //   // .style("text-anchor", "end ")
    //   .text(d3.format(".4f")((min+max)/2));

    // d3.select(window).on('resize', resize);





///////////////////////////////////////////////////////////////////////////
//////////////////// Create the Rainbow color gradient ////////////////////
///////////////////////////////////////////////////////////////////////////

//Calculate the gradient  
defs.append("linearGradient")
  .attr("id", "gradient-rainbow-colors")
  .attr("x1", "0%").attr("y1", "0%")
  .attr("x2", "100%").attr("y2", "0%")
  .selectAll("stop") 
  .data(coloursRainbow)                  
  .enter().append("stop") 
  .attr("offset", function(d,i) { return i/(coloursRainbow.length-1); })   
  .attr("stop-color", function(d) { return d; });

// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

     var mouseG = input_graphs['svg_input_' + name]
        .append("g")
        .attr("class", "mouse-over-effects" + name);


     var mouseG2 = output_graphs['svg_output_' + name] 
        .append("g")
        .attr("class", "mouse-over-effects2" + name);

     var mouseG3 = legend_graphs['legend_' + name]  
        .append("g")
        .attr("class", "mouse-over-effects3" + name);

      mouseG
        .append("path") // this is the black vertical line to follow mouse
        .attr("class", "mouse-line" + name)
        .style("stroke", "#393B45") //6E7889
        .style("stroke-width", "0.5px")
        .style("opacity", 0.75)

      mouseG2
        .append("path") // this is the black vertical line to follow mouse
        .attr("class", "mouse-line2" + name)
        .style("stroke", "#393B45") //6E7889
        .style("stroke-width", "0.5px")
        .style("opacity", 0.75)

      mouseG3
        .append("path") // this is the black vertical line to follow mouse
        .attr("class", "mouse-line3" + name)
        .style("stroke", "#393B45") //6E7889
        .style("stroke-width", "1px")
        .style("opacity", 0.75)

      mouseG.append("text")
        .attr("class", "mouse-text" + name)
        .attr('font-size', "8px")

      mouseG2.append("text")
        .attr("class", "mouse-text2" + name)

      mouseG3.append("text")
        .attr("class", "mouse-text3" + name)

      // var lines = document.getElementsByClassName('line');
      var lines = [path,path_out, path_out_true]

      // var totalLength = path.node().getTotalLength();
      var totalLength = 1000
      // var totalLength2 = path_out.node().getTotalLength();

      var mousePerLine = mouseG.selectAll('.mouse-per-line' + name)
        .data(values[name])
        .enter()
        .append("g")
        .attr("class", "mouse-per-line" + name);

      var mousePerLine2 = mouseG2.selectAll('.mouse-per-line2' + name)
        .data(values[name])
        .enter()
        .append("g")
        .attr("class", "mouse-per-line2" + name);

      var mousePerLine3 = mouseG2.selectAll('.mouse-per-line3' + name)
        .data(values[name])
        .enter()
        .append("g")
        .attr("class", "mouse-per-line3" + name);




      mousePerLine.append("circle")
        .attr("r", 4)
        .style("stroke", '#7c9ab0')
        .style("fill", "none")
        .style("stroke-width", "2px")
        .style("opacity", "0");

      mousePerLine2.append("circle")
        .attr("r", 4)
        .style("stroke", '#7c9ab0')
        .style("fill", "none")
        .style("stroke-width", "2px")
        .style("opacity", "0");

      // mousePerLine.append("text")
      //   .attr("transform", "translate(10,3)");

      // mousePerLine2.append("text")
      //   .attr("class", "valuetext");

      mouseG
        .append('svg:rect') // append a rect to catch mouse movements on canvas
        .attr('width', width) // can't catch mouse events on a g element
        .attr('height', height +100)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .attr('cursor', 'crosshair')
        .on('mouseout', function() { // on mouse out hide line, circles and text
          d3.select("#my_dataviz" + name + "3")
            .select(".mouse-line" + name)
            .style("opacity", "0")
          d3.select("#my_dataviz" + name + "4")
            .select(".mouse-line2" + name)
            .style("opacity", "0");
          d3.select("#legend" + name)
            .select(".mouse-line3" + name)
            .style("opacity", "0");
          d3.select("#my_dataviz" + name + "3")
            .select(".mouse-text" + name)
            .style("opacity", "0");
          d3.select("#my_dataviz" + name + "4")
            .select(".mouse-text2" + name)
            .style("opacity", "0");
          d3.select("#my_dataviz" + name + "3")
            .selectAll(".mouse-per-line" + name + "circle")
            .style("opacity", "0");
        })

        mouseG2
          .append('svg:rect') // append a rect to catch mouse movements on canvas
          .attr('width', width) // can't catch mouse events on a g element
          .attr('height', height) 
          .attr('fill', 'none')
          .attr('pointer-events', 'all')
          .attr('cursor', 'crosshair')
        .on('mouseout', function() {
          d3.select("#my_dataviz" + name + "4")
            .selectAll(".mouse-per-line2" + name + "circle")
            .style("opacity", "0"); })

        mouseG3
          .append('svg:rect') // append a rect to catch mouse movements on canvas
          .attr('width', width) // can't catch mouse events on a g element
          .attr('height', height) 
          .attr('fill', 'none')
          .attr('pointer-events', 'all')
          .attr('cursor', 'crosshair')
        .on('mouseout', function() {
          d3.select("#legend" + name)
            .selectAll(".mouse-per-line3" + name + "circle")
            .style("opacity", "0"); })

        const point = output_graphs['svg_output_' + name].append('circle')
          .attr('r', 4)
          .style("fill", 'none')
          .style("stroke", '#3A3B3C')
          .style("opacity", "0");

        const point_out2 = output_graphs['svg_output_' + name].append('circle')
          .attr('r', 4)
          .style("fill", 'none')
          .style("stroke", '#3A3B3C')
          .style("opacity", "0");

        const point_input = input_graphs['svg_input_' + name].append('circle')
          .attr('r', 4)
          .style("fill", 'none')
          .style("stroke", '#3A3B3C')
          .style("opacity", "0");

        const legend_indicator = legend_graphs['legend_' + name].append('polygon')
          // .attr('points', "0,0 10,5 0,10")
          // .style("fill", 'none')
          .style("stroke", '#3A3B3C')
          .style("opacity", "0");

        // make tooltip fo hovering over
        var tooltip = d3.select("#my_dataviz" + name + "2")
                        .append("div")
                        .style("opacity", 0)
                        .attr("class", "tooltip")
                        .style("background-color", "white")
                        .style("border", "solid")
                        .style("border-width", "1px")
                        .style("border-radius", "4px")
                        .style("padding", "4px")
                        .style("font-size", "11px")

      // Three function that change the tooltip when user hover / move / leave a cell
      var mouseover = function(d) {
        d3.select(this)
            .transition()
            .duration(5)     
            .style("fill-opacity", 0.3)
            // .style("stroke", "black")
            // .style("stroke-width", 1)
        d3.select("#my_dataviz" + name + "3")
            .select(".mouse-line" + name)
            .style("opacity", "1")
            .select(".mouse-text" + name)
            .style("opacity", "1")
            .select(".mouse-per-line" + name + "circle")
            .style("opacity", "1");
        d3.select("#my_dataviz" + name + "4")
            .select(".mouse-line2" + name)
            .style("opacity", "1")
            .select(".mouse-text2" + name)
            .style("opacity", "1")
            .select(".mouse-per-line2" + name + "circle")
            .style("opacity", "1")
        d3.select("#legend" + name)
            .select(".mouse-line3" + name)
            .style("opacity", "1")
            .select(".mouse-text3" + name)
            .style("opacity", "1")
        tooltip
            .style("opacity",0.75);


          var mouse = d3.mouse(this);
          d3.select("#my_dataviz" + name + "3")
            .select(".mouse-text" + name)
            .attr("x", mouse[0])
            .attr("transform", "translate(5," + ((height / 4) - 2) + ")")



          // d3.select("#my_dataviz3")
          //   .select(".mouse-line")
          //   .attr("d", function() {
          //     var d = "M" + mouse[0] + "," + height/4;
          //     d += " " + mouse[0] + "," + 0;
          //     return d;
          //   })

          // d3.select("#my_dataviz3")
          //   .selectAll(".mouse-per-line")
          //   .attr("transform", function(d, i) {
          //     if (i >= 2){ return null };

          //     var xDate = x_dates.invert(mouse[0])

          //     time = d3.timeFormat("%H:%M %p")(xDate)
              
          //     var beginning = 0,
          //         end = totalLength
          //         target = null;

          //     while (true){

          //       target = Math.floor((beginning + end) / 2);
          //       pos = path.node().getPointAtLength(target);

          //       if ((target === end || target === beginning) && pos.x !== mouse[0]) {
          //           break;
          //       }
          //       if (pos.x > mouse[0])      end = target;
          //       else if (pos.x < mouse[0]) beginning = target;

          //       else break; //position found
          //     }

          //     d3.select("#my_dataviz3").select('text')
          //       // .text(y.invert(pos.y).toFixed(1) + " MW") //xDate.toFixed(0) + ", "+
          //       .style("font", "11px arial");
          //     d3.select('#my_dataviz3').select('circle')
          //       .style("opacity", 1)
          //     var parseDate = d3.timeParse("%a %d");
          //     var timestamp = d3.select("#my_dataviz3").select('.mouse-text')
          //       .text(time)
          //       .style("opacity", 0.5)
          //       .style("text-transform", "uppercase")
          //       .style("font", "16px arial")


          //     return "translate(" + mouse[0] + "," + pos.y +")";

          //   });


          // heatmap.on('mousemove', () => {
          //     const x = d3.event.layerX;
          //     const curY = x_input.invert(x);
          //     console.log(x)

          //     const minY = Math.floor(curY);
          //     const maxY = Math.ceil(curY);
          //     if (data[minY] && data[maxY]) {
          //       const yDelta = curY - minY;
          //       const minP = data[minY].input_solar;
          //       const maxP = data[maxY].input_solar;
          //       const curP = minP + (maxP - minP) * yDelta;
          //       const xPos = y2(curP)

          //       point_input
          //         .attr('cx', x)
          //         .attr('cy', xPos)



                // d3.select("#my_dataviz3")
                //   .select(".mouse-line")
                //   .attr("d", function() {
                //     var d = "M" + y + "," + height/4;
                //     d += " " + y + "," + 0;
                //     return d;
                //   })

                // var xDate = x_dates.invert(y)
                // time = d3.timeFormat("%H:%M %p")(xDate)

                // d3.select("#my_dataviz3").select('.mouse-text')
                //   .text(time)
                //   .style("opacity", 0.5)
                //   .style("text-transform", "uppercase")
                //   .style("font", "16px arial")


              // }  
              
              // line_graph
              // y_test
            // }) 


          /////////////////////////////////////////////////// 
          d3.select("#my_dataviz" + name + "4")
          var mouse2 = d3.mouse(this);
           d3.select("#my_dataviz" + name + "4")
            .select(".mouse-text2" + name)
            .attr("y", mouse2[1])
            .attr("transform", "translate(" + (mouse2[1]+5) + "," + (mouse2[1]+2) + ") rotate(90)")

          // d3.select("#legend" + name)
          // var mouse3 = d3.mouse(this);
          //  d3.select("#legend" + name)
          //   .select(".mouse-text3" + name)
          //   .attr("y", mouse3[1])
          //   .attr("transform", "translate(" + (mouse3[1]) + "," + (mouse3[1]) + ") rotate(90)")

          // d3.select("#my_dataviz4")
          //   .select(".mouse-line2")
          //   .attr("d", function() {
          //     var d = "M" + height + "," + mouse2[1];
          //     d += " " + 0 + "," + mouse2[1];
          //     return d;
          //   })

          // d3.select("#my_dataviz4")
          //   .selectAll(".mouse-per-line2")
          //   .attr("transform", function(d, i) {
          //     if (i >= 2){ return null };
          //     var xDate2 = y_dates.invert(mouse2[1])

          //     time2 = d3.timeFormat("%H:%M %p")(xDate2)
          //     console.log('total-length-2')
          //     console.log(totalLength2)

              
          //     var beginning2 = 0,
          //         end2 = totalLength2
          //         target2 = null;

          //     while (true){

          //       target2 = Math.floor((beginning2 + end2) / 2);


          //       var pos2 = path3.node().getPointAtLength(target2);
          //       console.log(pos2.x)
          //       if ((target2 === end2 || target2 === beginning2) && pos2.y !== mouse2[1]) {
          //           break;
          //       }
          //       if (pos2.y > mouse2[1]) { end2 = target2; }
          //       else if (pos2.y < mouse2[1]) { beginning2 = target2; }

          //       else {break}; //position found
          //     }


          //     d3.select("#my_dataviz4").select('circle')
          //       .style("opacity", 1)

          //     var parseDate2 = d3.timeParse("%a %d");
          //     var timestamp2 = d3.select("#my_dataviz4").select('.mouse-text2')
          //       .text(time2)
          //       .style("opacity", 0.5)
          //       .style("text-transform", "uppercase")
          //       .style("font", "16px arial")



          //     return "translate(" + (pos2.x) + "," + mouse2[1] +")";

          //   });



          heatmap_graphs_2['heatmap' + name].on('mousemove', () => {

              const x = d3.event.layerX + margin.left + 10; 
              const curX = scales['xscale'].invert(x);
              const minX = Math.floor(curX);
              const maxX = Math.ceil(curX);
              if (values[name][minX] && values[name][maxX]) {
                const xDelta = curX - minX;
                const minPx = values[name][minX].input_solar;
                const maxPx = values[name][maxX].input_solar;
                const curPx = minPx + (maxPx - minPx) * xDelta;
                const yPos = scales['yscale'](curPx)

                point_input
                  .attr('cx', x)
                  .attr('cy', yPos)
                  .style("opacity", "1");

                d3.select("#my_dataviz" + name + "3")
                  .select(".mouse-line" + name)
                  .attr("d", function() {
                      var d = "M" + x + "," + height/4;
                      d += " " + x + "," + 0;
                    return d;
                  })

                var xDate = scale_dates['xscale'].invert(x)
                time = d3.timeFormat("%H:%M %p")(xDate)

                d3.select("#my_dataviz" + name + "3")
                  .select('.mouse-text' + name)
                  .text(time)
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "13px arial")

                // tooltip.html(curPx)
                //   .style("left", (d3.mouse(this)[0]+70) + "px")
                //   .style("top", (d3.mouse(this)[1]) + "px")

              

              const y = d3.event.layerY;
              const curY = scales_out['yscale'].invert(y);
              const minY = Math.floor(curY);
              const maxY = Math.ceil(curY);
              const legend_loc = legend_scales['legendScale' + name](d.value);
              if (values[name][minY] && values[name][maxY]) {
                const yDelta = curY - minY;
                const minP = values[name][minY].prediction;
                const maxP = values[name][maxY].prediction;
                const curP = minP + (maxP - minP) * yDelta;
                const xPos = scales_out['xscale'](curP)
                // get true value point location
                const minP_true = values[name][minY].y_true;
                const maxP_true = values[name][maxY].y_true;
                const curP_true = minP_true + (maxP_true - minP_true) * yDelta;
                const xPos_true = scales_out['xscale'](curP_true)

                // console.log(xPos);
                point
                    .attr('cx', xPos)
                    .attr('cy', y)
                    .style("opacity", "1");

                point_out2
                    .attr('cx', xPos_true)
                    .attr('cy', y)
                    .style("opacity", "1");

                d3.select("#my_dataviz" + name + "4")
                  .select(".mouse-line2" + name)
                  .attr("d", function() {
                    var d = "M" + (output_graph_width+30) + "," + y;
                    d += " " + 0 + "," + y;
                    return d;
                  })


                d3.select("#legend" + name)
                var mouse3 = d3.mouse(this);

                if (width < 700) {
                 d3.select("#legend" + name)
                  .select(".mouse-text3" + name)
                  .attr("y", legend_loc)
                  .attr("transform", "translate(" + (0) + "," + (-10) + ")")
                } else {
                 d3.select("#legend" + name)
                  .select(".mouse-text3" + name)
                  .attr("y", legend_loc)
                  .attr("transform", "translate(" + (-30.5) + "," + (2.5) + ")")      
                }

                d3.select("#legend" + name)
                  .select(".mouse-line3" + name)
                  .attr("d", function() {
                    var d = "M" + 18 + "," + legend_loc;
                    d += " " + 0 + "," + legend_loc;
                    return d;
                  })

              legend_indicator
                .attr('points',  "0," + (legend_loc-5) + " " + "10," + (legend_loc) + " " + "0,"+ (legend_loc+5))
                // .attr('points', "0,-5 10,0 0,5")
                // .attr('points', "0,10 10,15 0,20")
                .style("opacity", "1");


                var xDate2 = scale_dates['yscale'].invert(y)
                time2 = d3.timeFormat("%H:%M %p")(xDate2)

                d3.select("#my_dataviz" + name + "4").select('.mouse-text2' + name)
                  .text(time2)
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "13px arial")


                d3.select("#legend" + name).select('.mouse-text3' + name)
                  .text(d.value.toFixed(4))
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "9px arial")


                d3.select("#my_dataviz" + name + "2").select('.tooltip')
                // tooltip
                  .html("<strong>Context:&nbsp</strong>" + d.value.toFixed(4) + "<br><strong>Input:&nbsp</strong>" + curPx.toFixed(4) + "MW<br><strong>Prediction:&nbsp</strong>" + curP.toFixed(4)  + "MW" + "<br><strong>Actual:&nbsp</strong>" + curP_true.toFixed(4)  + "MW")
                  .style("left", (d3.mouse(this)[0]+40) + "px")
                  .style("top", (d3.mouse(this)[1]) + "px")
                  .attr('cursor', 'crosshair')  


              }  }

              // line_graph
              // y_test
            }) 

            /////////////////////////////////////////////////// 

      }
       // d3.select("#my_dataviz4").selectAll('svg').attr('transform', 'translate(-10,-100) rotate(-40)')
      var mouseleave = function(d) {
        d3.select(this)
           .transition()
           .duration(750)
           .style("fill-opacity", 1);
        d3.select("#my_dataviz" + name + "3")
          .select(".mouse-line" + name)
          .style("opacity", "0")
        d3.select("#my_dataviz" + name + "3")
          .select('.mouse-text' + name)
          .style("opacity", "0")
        d3.select("#my_dataviz" + name + "4")
          .select(".mouse-line2" + name)
          .style("opacity", "0") 
        d3.select("#my_dataviz" + name + "4")
          .select('.mouse-text2' + name)
          .style("opacity", "0")          
        point.style("opacity", "0")
        point_out2.style("opacity", "0")
        point_input.style("opacity", "0")
        tooltip.style("opacity", 0)

        ;

      }


      heatmap_graphs_2['heatmap' + name].style("fill", function(d) { return colorScaleRainbow(colorInterpolateRainbow(d.value_scaled)); } )
          .on("mouseover", mouseover)
          // .on("mousemove", mousemove)
          .on("mouseleave", mouseleave)



    


  
    })
  

  input_scales.push(scales)
  input_axes.push(axes)

  heatmap_scales.push(scales_heat)
  heatmap_axes.push(axes_heat)

  dates_scale.push(scale_dates)
  dates_axes.push(axes_dates)
  dates_min_axes.push(axes_dates_min)
  dates_noon_axes.push(axes_dates_noon)

  output_scales.push(scales_out)
  output_axes.push(axes_out)


  d3.select(window).on('resize',resize);
  


}


 var resize = function(e) {

      var width = parseInt(d3.select('.slideshow-container').style('width'), 10) + 150

      console.log(width)

      // grab names 
      for (var idx = 0; idx < (Object.keys(input_graphs).length); idx++) {
        // update svg input chart /////////////////////////////////////////////////////////
        current_chart = charts[idx]

        
        if (width < 700) { 
          height = 210
        } else {
          height = 250
        } 

        dimensions = width_check(width, height)
        main_graph_width = dimensions[0]
        buffer = dimensions[1]
        output_graph_width = dimensions[2]

      
        // d3.select("#my_dataviz" + name + "3").select("svg").attr("width", width)
        d3.select("#my_dataviz" + current_chart + "3").select("svg").attr("width", width)
        input_graphs[Object.keys(input_graphs)[idx]].select("svg").attr("width", width)
        input_scales[idx]['xscale'].range([0, main_graph_width]);
        // input_scales[idx]['yscale'].range([height/4, 0]);


        input_axes[idx]['xAxis'].scale(input_scales[idx]['xscale']);
        // input_axes[idx]['yAxis'].scale(input_scales[idx]['yscale']);

        // input_groups['svg_input_Xaxis' + current_chart].call(input_axes[idx]['xAxis'])

        // input_graphs[Object.keys(input_graphs)[idx]].call(input_axes[idx]['xAxis'])
        // input_graphs[Object.keys(input_graphs)[idx]].call(input_axes[idx]['yAxis'])

        input_graphs[Object.keys(input_graphs)[idx]].select(".x.axis" + current_chart).call(input_axes[idx]['xAxis'])
        input_graphs[Object.keys(input_graphs)[idx]].select('.line' + current_chart).attr("d", input_lines[current_chart])

      



      // update svg heatmap chart /////////////////////////////////////////////////////////

        // d3.select("#my_dataviz" + name + "2").select("svg").attr("width", width)
        d3.select("#my_dataviz" + current_chart + "2").selectAll("svg").attr("width", width)
        heatmap_graphs[Object.keys(heatmap_graphs)[idx]].select("svg").attr("width", width)

        heatmap_scales[idx]['xscale'].range([0, main_graph_width]);
        // heatmap_scales[idx]['yscale'].range([height, 0]);

        heatmap_axes[idx]['xAxis'].scale(heatmap_scales[idx]['xscale']);
        // // heatmap_axes[idx]['yAxis'].scale(heatmap_scales[idx]['yscale']);

        heatmap_groups['svg_heat_Xaxis' + current_chart].call(heatmap_axes[idx]['xAxis'])

        // heatmap_graphs[Object.keys(heatmap_graphs)[idx]].call(heatmap_axes[idx]['xAxis'])
        // // heatmap_graphs[Object.keys(heatmap_graphs)[idx]].call(heatmap_axes[idx]['yAxis'])

        heatmap_graphs[Object.keys(heatmap_graphs)[idx]].select(".x.axis" + current_chart).call(heatmap_axes[idx]['xAxis'])


        heatmap_graphs_2['heatmap' + current_chart] .attr("x", function(d) { return heatmap_scales[idx]['xscale'](d.group) })
          .attr("y", function(d) { return heatmap_scales[idx]['yscale'](d.variable )})
          .attr("width", heatmap_scales[idx]['xscale'].bandwidth()+0.6)
          .attr("height", heatmap_scales[idx]['yscale'].bandwidth()+0.6)


        //  update dates scales
        dates_scale[idx]['xscale'].range([0, main_graph_width]);
        // dates_scale[idx]['yscale'].range([height, 0]);

        dates_axes[idx]['xAxis'].scale(dates_scale[idx]['xscale']);
        // dates_axes[idx]['yAxis'].scale(dates_scale[idx]['yscale']);

        dates_min_axes[idx]['xAxis'].scale(dates_scale[idx]['xscale']);
        // dates_min_axes[idx]['yAxis'].scale(dates_scale[idx]['yscale']);

        dates_noon_axes[idx]['xAxis'].scale(dates_scale[idx]['xscale']);

        axis_date_groups['svg_heat_Xdates' + current_chart].call(dates_axes[idx]['xAxis'])
        // axis_date_groups['svg_heat_Ydates' + current_chart].call(dates_axes[idx]['yAxis'])

        axis_date_groups['svg_heat_min_ticks' + current_chart].call(dates_min_axes[idx]['xAxis'])
        // axis_date_groups['svg_heat_noon_ticks' + current_chart].call(dates_min_axes[idx]['xAxis'])

        heatmap_graphs['heatmap_' + current_chart].select(".x.axis_dates" + current_chart).call(dates_axes[idx]['xAxis'])
        heatmap_graphs['heatmap_' + current_chart].select(".x.axis_dates_min" + current_chart).call(dates_min_axes[idx]['xAxis'])
        heatmap_graphs['heatmap_' + current_chart].select(".x.axis_dates_noon" + current_chart).call(dates_noon_axes[idx]['xAxis'])


        // update svg output chart /////////////////////////////////////////////////////////
        d3.select("#my_dataviz" + current_chart + "4").select("svg").attr("width", width)
        // d3.select("#my_dataviz" + name + "4").selectAll("svg").attr("width", width)
        output_graphs[Object.keys(output_graphs)[idx]].select("svg").attr("width", width)

        output_scales[idx]['xscale'].range([0, output_graph_width]);
        output_scales[idx]['yscale'].range([height, 0]);

        output_axes[idx]['xAxis'].scale(output_scales[idx]['xscale']);
        output_axes[idx]['yAxis'].scale(output_scales[idx]['yscale']);

        output_graphs[Object.keys(output_graphs)[idx]].select(".x.axis_out" + current_chart).call(output_axes[idx]['xAxis']).select(".domain").remove();
        output_graphs[Object.keys(output_graphs)[idx]].select('.line_output' + current_chart).attr("d", output_lines[Object.keys(output_lines)[idx]](values[current_chart]))
        output_graphs[Object.keys(output_graphs)[idx]].select('.line_output_true' + current_chart).attr("d", output_line_true[Object.keys(output_line_true)[idx]](values[current_chart]))

        var adj_width = width - buffer

        output_graphs[Object.keys(output_graphs)[idx]]
          .attr("transform",
            "translate(" + adj_width + "," + 0 + ")");

        output_graphs[Object.keys(output_graphs)[idx]].select(".output_svg_back" + current_chart).attr('width', width)



        }








      // // update legend /////////////////////////////////////////////////////////

      // // get legend width 
      // legend_vars = legend_size(main_graph_width)
      // legend_width = legend_vars[0]
      // legend_start =  legend_vars[1]


      // d3.select("#legend" + name).select("svg").attr("width", width)
      // legend.attr("width", width)

      // legendScale.range([legend_start, legend_width])
      // legendAxis.scale(legendScale)
      // legend_axis_ref.call(legendAxis)
      // legend.select(".x.axis_legend" + name).call(legendScale)
      // legend.select("rect")
      //   .attr("width", legend_width - legend_start)
      //   .attr("transform", "translate(" + legend_start + ",0)")


      // legend.select(".legend_max_txt" + name)
      //   .attr("x", (main_graph_width - legend_start - 20))

      // legend.select(".legend_min_txt" + name)
      //   .attr("x", legend_start - 10)

      // legend.select(".legend_mid_txt" + name)
      //   .attr("x", (main_graph_width/2)-10)




      // // update svg output chart /////////////////////////////////////////////////////////



      // svg_output_Xaxis.call(xAxis_out)
      // svg_output_Yaxis.call(yAxis_out)

      // svg_output.select(".x.axis_out" + name).call(xAxis_out).select(".domain").remove();
      // // svg_output.select(".y.axis_out").call(yAxis_out)

      // svg_output.select('.line_output' + name).attr("d", output_line(data_output))


      // var adj_width = width - buffer
      // // svg_output.attr("width", width)
      // svg_output
      //   // .attr('width', width + 100)+
      //   .attr("transform",
      //     "translate(" + adj_width + "," + 0 + ")");

      // svg_output.select(".output_svg_back" + name).attr('width', width)


      // special width resize for mobile screens ///////////////////////////////////////////
      // if (width <= 700) {
      //   output_graph_width = 
      // }


      }




// helper function to make dynamic legend that varies with screen size
function legend_size(current_width) {


  if (700 <= current_width && current_width <= 800) {

    legend_width = current_width - 100 
    legend_start = 100
  }
  else if (current_width < 700) {

    legend_width = current_width - 50
    legend_start = 50
  }
  else {

    legend_width = current_width - 200 // -550 for rectangle
    legend_start = 200 
  }


  return [legend_width, legend_start]
} 


// helper function to ammend width of graphs for mobile cause
function width_check(current_width, current_height) {

  if (current_width <= 700) { 
    output_graph_width = current_height / 4
    buffer = 75
    main_graph_width = current_width - 112.5

  }
  else {
    output_graph_width = (current_height / 3.75)
    buffer = 120
    main_graph_width = current_width - 160    
  }


  return [main_graph_width, buffer, output_graph_width]
}
















