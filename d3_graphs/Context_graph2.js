
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


// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


width = parseInt(d3.select('#content_graphs').style('width'), 10) 

function context_graph(context_data, output_data, name) {

// function instaniate_charts() {
  // set the dimensions and margins of the graph
  var margin = {top: 0, right: 0, bottom: 0, left: 25},
  // width = 1500 
  height = 275


  // append the svg object to the body of the page
  var svg_heatmap = d3.select("#my_dataviz" + name + "2")
  .append("svg")
    .attr("width", width)
    .attr("height", height)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

  // heatmap legend
  var legend = d3.select('#legend')
  .append("svg")
    .attr("width", width)
    .attr("height", height -210)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + 0+ ")");



  // input line graph
  var svg_input = d3.select("#my_dataviz" + name + "3")
  .append("svg")
    .attr("width", width)
    .attr("height", height - 190)   
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


    // Add output line graph
    var svg_output = d3.select("#my_dataviz" + name + "4")
    .append("svg")
      .attr("class", 'output_svg_back' + name)
      .attr("width", width)
      .attr("height", height)
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
  svg_input.append("text")
          .attr("x", (width /2)-margin.right)             
          .attr("y", 10)
          .style("font-size", "16px") 
          .style("text-decoration", "underline")  
          .text(name);


  // update current width of graphs
  width = parseInt(d3.select('#content_graphs').style('width'), 10) 
  height = 210

  dimensions = width_check(width, height)
  main_graph_width = dimensions[0]
  buffer = dimensions[1]
  output_graph_width = dimensions[2]

  
  console.log(width)
  var adj_width = width - buffer

  svg_output
  .attr("transform",
      "translate(" + adj_width + "," + 0 + ")");

// INPUT LINE SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////

var x_input = d3.scaleLinear()
  .range([0, main_graph_width])
  // .domain([0, 100])

var y_input = d3.scaleLinear()
  // .domain([0, 0.5])
  .range([ height/4, 0 ]);

var xAxis = d3.axisBottom().scale(x_input).tickSizeOuter(0).tickValues([]);
var yAxis = d3.axisLeft().scale(y_input).tickSizeOuter(0).tickValues([]);

var input_line = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return x_input(d.input_time_ref) })
                .y(function(d) { return y_input(d.input_solar) })


// HEATMAP SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////


x_heat = d3.scaleBand()
  .range([0, main_graph_width])

y_heat = d3.scaleBand()
  .range([height, 0])

xAxis_heat = d3.axisBottom().scale(x_heat).tickSizeOuter(0).tickValues([]);
yAxis_heat = d3.axisLeft().scale(y_heat).tickSizeOuter(0).tickValues([]);


x_dates = d3.scaleTime()
  .range([ 0, main_graph_width])

y_dates = d3.scaleTime()
  .range([ height, 0 ])

xAxis_dates = d3.axisBottom().scale(x_dates).tickFormat(d3.timeFormat("")).ticks(d3.timeHour.every(24)).tickSize(16);
yAxis_dates = d3.axisLeft().scale(y_dates).tickFormat(d3.timeFormat("%a %d")).ticks(d3.timeHour.every(12)).tickSize(8); // Noon y-tick




// axis for minor tick marks 
x_min_ticks = d3.scaleTime()
  .range([0, main_graph_width])

xAxis_minor_ticks = d3.axisBottom().scale(x_dates).ticks(96).tickFormat('').tickSize(3)
yAxis_minor_ticks = d3.axisLeft().scale(y_dates).ticks(24).tickFormat('').tickSize(3)

xAxis_noon_ticks = d3.axisBottom().scale(x_dates).tickFormat(d3.timeFormat("%a %d")).ticks(d3.timeHour.every(12)).tickSize(8).tickPadding(8);



// get width of legend relative to screen size
legend_vars = legend_size(main_graph_width)
legend_width = legend_vars[0]
legend_start =  legend_vars[1]

// setup for legend
legendScale = d3.scaleLinear()
   .range([legend_start, legend_width])

//Set up X axis
legendAxis = d3.axisBottom().scale(legendScale)


// OUTPUT SETUP ///////////////////////////////////////////////////////////////////////////////////////////////////////////

x_output = d3.scaleLinear()
  .range([ 0, output_graph_width ]);
  // .domain([0, 100])

y_output = d3.scaleLinear()
  // .domain([0, 0.5])
  .range([ height, 0 ]);

xAxis_out = d3.axisBottom().scale(x_output).tickSizeOuter(0).tickValues([]);
yAxis_out = d3.axisLeft().scale(y_output).tickSizeOuter(0).tickValues([]);

output_line = d3.line()
                .curve(d3.curveBasis)
                .x(function(d2) { return x_output(d2.prediction) })
                .y(function(d2) { return y_output(d2.output_time) })




// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Get the data
d3.csv(context_data, 

    function(d){
    return { group_index: d.group_index = +d.group_index,
             variable_index: d.variable_index = +d.variable_index,
             group: d3.timeParse("%d/%m/%Y %H:%M")(d.group), 
             variable: d3.timeParse("%d/%m/%Y %H:%M")(d.variable),
             value: d.value = +d.value,
             input_time_ref: d.input_time_ref = +d.input_time_ref,
             input_time: d3.timeParse("%d/%m/%Y %H:%M")(d.input_time),
             input_solar: d.input_solar = +d.input_solar,
             input_cloud_cover: d.input_cloud_cover = +d.input_cloud_cover,
             // output_time_ref: d.output_time_ref = +d.output_time_ref,
             // output_time: d3.timeParse("%d/%m/%Y %H:%M")(d.output_time),
             prediction: d.prediction = +d.prediction,
    }
  },



  function(data) {

    d3.csv(output_data, 

    function(d2){
    return { output_time: d2.output_time = +d2.output_time,
             prediction: d2.prediction = +d2.prediction,
    }
  },


  function(data_out) {



    values =  data 
    data_output = data_out

    var defs = svg_heatmap.append("defs");
    color = ["#2c7bb6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c", "#f9d057","#f29e2e","#e76818","#d7191c"]
    var colorScale = d3.scaleLinear().range(color);

    // List of groups = header of the csv files
    var keys = data.columns.slice(1);

// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HEATMAP GRAPH

    x_heat.domain(d3.map(data, function(d){return d.group;}).keys())

    var svg_heat_Xaxis = svg_heatmap.append("g")
      .attr("class", "x axis" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis_heat)
      .select(".domain").remove()

    y_heat.domain(d3.map(data, function(d){return d.variable;}).keys())
    svg_heat_Yaxis = svg_heatmap.append("g")
      .attr("class", "y axis" + name)
      .call(yAxis_heat)



var heatmap = svg_heatmap.selectAll()
    .data(data, function(d) {return d.group+':'+d.variable;})
    .enter()
    .append("rect")
      .attr("x", function(d) { return x_heat(d.group) })
      .attr("y", function(d) { return y_heat(d.variable )})
      .attr("width", x_heat.bandwidth())
      .attr("height", y_heat.bandwidth())
      // .style("fill", function(d) { return colorScaleRainbow(colorInterpolateRainbow(d.value)); } )
      .style("opacity", 1.0)
      .style("stroke-width", 0)
      .style("stroke", "none")


    // Build color scale
    var myColor = d3.scaleSequential()
      .domain(d3.extent(data, function(d){return d.value;}))
      .interpolator(color);

      //scale the colors
    colorScale.domain(d3.extent(data,function(d){
      return d.value;
    }));


    x_dates.domain(d3.extent(data, function(d){return d.group;}))

    var svg_heat_Xdates = svg_heatmap.append("g")
      .attr("class", "x axis_dates" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis_dates)

    y_dates.domain(d3.extent(data, function(d){return d.variable;}))
    
    var svg_heat_Ydates = svg_heatmap.append("g")
      .style("text-transform", "uppercase")
      .style("font-size", "10px")
      .call(yAxis_dates)
    .selectAll('text')
      .attr("transform", "translate(-20, -20) rotate(-90)")


    var svg_heat_Ydates_min = svg_heatmap.append("g")
      .call(yAxis_minor_ticks)
      .select(".domain").remove();


    var svg_heat_noon_ticks = svg_heatmap.append("g")
      .attr("class", "x axis_dates_noon" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis_noon_ticks)
      .style("text-transform", "uppercase")
      .style("font-size", "10px")
      .select(".domain").remove();


    var svg_heat_min_ticks = svg_heatmap.append("g")
      .attr("class", "x axis_date_min" + name)
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis_minor_ticks)
      // .select(".domain").remove();


    // tick value on every second value
    var ticks = d3.select("#my_dataviz" + name + "2").selectAll(".tick text");
    ticks.each(function(_,i){
        if(i%2 == 0) d3.select(this).remove();
    });


// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// INPUT GRAPH

    x_input.domain([0, d3.max(data, function(d) { return +d.input_time_ref; })])
    y_input.domain([0, d3.max(data, function(d) { return +d.input_solar; })])

    var svg_input_Xaxis = svg_input.append("g")
      .attr("class", "x axis" + name)
      .attr("transform", "translate(0," + height/4 + ")")
      .call(xAxis);

      // .select(".domain").remove();

    var svg_input_Yaxis = svg_input.append("g")
      .attr("class", "y axis" + name)
      .call(yAxis)
      .select(".domain").remove();

    // Add the line path.
    var path = svg_input.append("path")
      .datum(values)
      .attr("class", "line" + name)
      .attr("fill", "steelblue")
      .attr("fill-opacity", 0.2)
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1)
      .attr("d", input_line(values));

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


    x_output.domain([0, d3.max(data_out, function(d2) { return +d2.prediction; })])
    y_output.domain([0, d3.max(data_out, function(d2) { return +d2.output_time; })])

    var svg_output_Xaxis = svg_output.append("g")
      .attr("class", "x axis_out" + name)
      .attr("transform", "translate(" + 0 + "," + height + ")")
      .call(xAxis_out)
      .select(".domain").remove();

    var svg_output_Yaxis = svg_output.append("g")
      .attr("class", "y axis_out" + name)
      .call(yAxis_out)
      // .select(".domain").remove();

    // Add the line path.
    var path_out = svg_output.append("path")
      .datum(data_out)
      .attr("class", "line_output" + name)
      .attr("fill", "steelblue")
      .attr("fill-opacity", 0.2)
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1)
      .attr("d", output_line(data_out));






///////////////////////////////////////////////////////////////////////////
//////////// Get continuous color scale for the Rainbow ///////////////////
///////////////////////////////////////////////////////////////////////////

var coloursRainbow = ["#2c7bb6", "#00a6ca","#00ccbc","#90eb9d","#ffff8c","#f9d057","#f29e2e","#e76818","#d7191c"];
var colourRangeRainbow = d3.range(0, 1, 1.0 / (coloursRainbow.length - 1));
colourRangeRainbow.push(1);
       
//Create color gradient
var colorScaleRainbow = d3.scaleLinear()
  .domain(colourRangeRainbow)
  .range(coloursRainbow)
  .interpolate(d3.interpolateHcl);

//Needed to map the values of the dataset to the color scale
var colorInterpolateRainbow = d3.scaleLinear()
  .domain(d3.extent(data, function(d){return d.value;}))
  .range([0,1]);

// get min and max of the dataset
var min = d3.min(data, function(d){return d.value;})
var max = d3.max(data, function(d){return d.value;})


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

     var mouseG = svg_input
        .append("g")
        .attr("class", "mouse-over-effects" + name);

     var mouseG2 = svg_output
        .append("g")
        .attr("class", "mouse-over-effects2" + name);

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

      mouseG.append("text")
        .attr("class", "mouse-text" + name)
        .attr('font-size', "8px")

      mouseG2.append("text")
        .attr("class", "mouse-text2" + name)


      // var lines = document.getElementsByClassName('line');
      var lines = [path,path_out]

      // var totalLength = path.node().getTotalLength();
      var totalLength = 1000
      var totalLength2 = path_out.node().getTotalLength();

      var mousePerLine = mouseG.selectAll('.mouse-per-line' + name)
        .data(data)
        .enter()
        .append("g")
        .attr("class", "mouse-per-line" + name);

      var mousePerLine2 = mouseG2.selectAll('.mouse-per-line2' + name)
        .data(data_out)
        .enter()
        .append("g")
        .attr("class", "mouse-per-line2" + name);



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
        .attr('height', height)
        .attr('fill', 'none')
        .attr('pointer-events', 'all')
        .on('mouseout', function() { // on mouse out hide line, circles and text
          d3.select("#my_dataviz" + name + "3")
            .select(".mouse-line" + name)
            .style("opacity", "0")
          d3.select("#my_dataviz" + name + "4")
            .select(".mouse-line2" + name)
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
        .on('mouseout', function() {
          d3.select("#my_dataviz" + name + "4")
            .selectAll(".mouse-per-line2" + name + "circle")
            .style("opacity", "0"); })

        const point = svg_output.append('circle')
          .attr('r', 4)
          .style("fill", 'none')
          .style("stroke", '#3A3B3C')
          .style("opacity", "0");

        const point_input = svg_input.append('circle')
          .attr('r', 4)
          .style("fill", 'none')
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



          heatmap.on('mousemove', () => {

              const x = d3.event.layerX - margin.left;
              const curX = x_input.invert(x);
              const minX = Math.floor(curX);
              const maxX = Math.ceil(curX);
              if (data[minX] && data[maxX]) {
                const xDelta = curX - minX;
                const minPx = data[minX].input_solar;
                const maxPx = data[maxX].input_solar;
                const curPx = minPx + (maxPx - minPx) * xDelta;
                const yPos = y_input(curPx)

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

                var xDate = x_dates.invert(x)
                time = d3.timeFormat("%H:%M %p")(xDate)

                d3.select("#my_dataviz" + name + "3").select('.mouse-text' + name)
                  .text(time)
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "13px arial")

                // tooltip.html(curPx)
                //   .style("left", (d3.mouse(this)[0]+70) + "px")
                //   .style("top", (d3.mouse(this)[1]) + "px")

              

              const y = d3.event.layerY - 10;
              const curY = y_output.invert(y);
              const minY = Math.floor(curY);
              const maxY = Math.ceil(curY);
              if (data_out[minY] && data_out[maxY]) {
                const yDelta = curY - minY;
                const minP = data_out[minY].prediction;
                const maxP = data_out[maxY].prediction;
                const curP = minP + (maxP - minP) * yDelta;
                const xPos = x_output(curP)
                // console.log(xPos);
                point
                  .attr('cx', xPos)
                  .attr('cy', y)
                  .style("opacity", "1");

                d3.select("#my_dataviz" + name + "4")
                  .select(".mouse-line2" + name)
                  .attr("d", function() {
                    var d = "M" + output_graph_width + "," + y;
                    d += " " + 0 + "," + y;
                    return d;
                  })

                var xDate2 = y_dates.invert(y)
                time2 = d3.timeFormat("%H:%M %p")(xDate2)

                d3.select("#my_dataviz" + name + "4").select('.mouse-text2' + name)
                  .text(time2)
                  .style("opacity", 0.5)
                  .style("text-transform", "uppercase")
                  .style("font", "13px arial")

                d3.select("#my_dataviz" + name + "2").select('.tooltip')
                // tooltip
                  .html("<strong>Context:&nbsp</strong>" + d.value.toFixed(4) + "<br><strong>Input:&nbsp</strong>" + curPx.toFixed(4) + "MW<br><strong>Output:&nbsp</strong>" + curP.toFixed(4)  + "MW")
                  .style("left", (d3.mouse(this)[0]+70) + "px")
                  .style("top", (d3.mouse(this)[1]) + "px")


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
        point_input.style("opacity", "0")
        tooltip.style("opacity", 0)

        ;

      }


      heatmap.style("fill", function(d) { return colorScaleRainbow(colorInterpolateRainbow(d.value)); } )
          .on("mouseover", mouseover)
          // .on("mousemove", mousemove)
          .on("mouseleave", mouseleave)



      legendWidth = width - 200
      legendHeight = 7.5


      //  make the heatmap legend
      // https://www.visualcinnamon.com/2016/05/smooth-color-legend-d3-svg-gradient/
      var defs = legend.append("defs");

      //Append a linearGradient element to the defs and give it a unique id
      var linearGradient = defs.append("linearGradient")
          .attr("id", "linear-gradient");


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
      legend.append("rect")
          .attr("width", legend_width - legend_start)
          .attr("height", legendHeight)
          .attr("transform", "translate(" + legend_start + ",0)")
          .style("fill", "url(#linear-gradient)");

      //Append title
      // legend.append("text")
      //   .attr("class", "legendTitle")
      //   .attr("x", 100)
      //   .attr("y", -1)
      //   .style("text-anchor", "middle")
      //   .style("fill", "#4F4F4F")
      //   .text("Context");

      legendScale.domain([min,max])

      legend_axis_ref = legend.append("g")
        .attr("class", "x axis_legend")
        .attr("transform", "translate(" + (-legendWidth/legendWidth+1) + "," + (legendHeight) + ")")
        .call(legendAxis.ticks(0).tickSize(0))
        .select(".domain").remove();

      // manually append min and max to legend
      legend
        .append("text")
        .attr("class", "legend_max_txt" + name)
        .attr("x", (main_graph_width - legend_start - 20))
        .attr("y", 18)
        .attr("font-family", "arial")
        .attr("font-size", "10.25")
        // .style("text-anchor", "end ")
        .text(d3.format(".4f")(max));

      legend
        .append("text")
        .attr("class", "legend_min_txt" + name)
        .attr("x", legend_start - 10)
        .attr("y", 18)
        .attr("font-family", "arial")
        .attr("font-size", "10.25")
        // .style("text-anchor", "end ")
        .text(d3.format(".4f")(min));

      legend
        .append("text")
        .attr("class", "legend_mid_txt" + name)
        .attr("x",(main_graph_width/2)-10)
        .attr("y", 18)
        .attr("font-family", "arial")
        .attr("font-size", "10.25")
        // .style("text-anchor", "end ")
        .text(d3.format(".4f")((min+max)/2));

      // d3.select(window).on('resize', resize);


  })
    })

  d3.select(window).on('resize',resize);
  return width

}


 var resize = function(e) {

      // console.log(name)

      // update svg input chart /////////////////////////////////////////////////////////
      width = parseInt(d3.select('#content_graphs').style('width'), 10)
      height = 210

      dimensions = width_check(width, height)
      main_graph_width = dimensions[0]
      buffer = dimensions[1]
      output_graph_width = dimensions[2]


      d3.select("#my_dataviz" + name + "3").select("svg").attr("width", width)
      svg_input.attr("width", width)
      x_input.range([0, main_graph_width]);
      y_input.range([height/4, 0]);

      xAxis.scale(x_input);
      yAxis.scale(y_input);

      svg_input_Xaxis.call(xAxis)
      svg_input_Yaxis.call(yAxis)

      svg_input.select(".x.axis" + name).call(xAxis)

      svg_input.select('.line' + name).attr("d", input_line(values))



      // update svg heatmap chart /////////////////////////////////////////////////////////

      // d3.select("#my_dataviz2").select("svg").attr("width", width)
      d3.select("#my_dataviz" + name + "2").selectAll("svg").attr("width", width)
      svg_heatmap.attr("width", width)

      x_heat.range([0, main_graph_width]);
      y_heat.range([height, 0]);

      xAxis_heat.scale(x_heat);
      // yAxis_heat.scale(y_heat);

      svg_heat_Xaxis.call(xAxis_heat)
      // svg_heat_Yaxis.call(yAxis_heat)

      svg_heatmap.select(".x.axis" + name).call(xAxis_heat)


      heatmap.attr("x", function(d) { return x_heat(d.group) })
        .attr("y", function(d) { return y_heat(d.variable )})
        .attr("width", x_heat.bandwidth())
        .attr("height", y_heat.bandwidth())


      x_dates.range([0, main_graph_width]);
      // y_dates.range([height, 0]);

      xAxis_dates.scale(x_dates);
      // yAxis_dates.scale(y_dates);

      xAxis_minor_ticks.scale(x_dates);
      xAxis_noon_ticks.scale(x_dates);

      svg_heat_Xdates.call(xAxis_dates)
      // svg_heat_Ydates.call(yAxis_dates)

      svg_heat_min_ticks.call(xAxis_minor_ticks)
      svg_heat_noon_ticks.call(xAxis_noon_ticks)

      svg_heatmap.select(".x.axis_dates" + name).call(xAxis_dates)
      svg_heatmap.select(".x.axis_dates_min" + name).call(xAxis_minor_ticks)
      svg_heatmap.select(".x.axis_dates_noon" + name).call(xAxis_noon_ticks)


      // get legend width 
      legend_vars = legend_size(main_graph_width)
      legend_width = legend_vars[0]
      legend_start =  legend_vars[1]


      d3.select("#legend" + name).select("svg").attr("width", width)
      legend.attr("width", width)

      legendScale.range([legend_start, legend_width])
      legendAxis.scale(legendScale)
      legend_axis_ref.call(legendAxis)
      legend.select(".x.axis_legend" + name).call(legendScale)
      legend.select("rect")
        .attr("width", legend_width - legend_start)
        .attr("transform", "translate(" + legend_start + ",0)")


      legend.select(".legend_max_txt" + name)
        .attr("x", (main_graph_width - legend_start - 20))

      legend.select(".legend_min_txt" + name)
        .attr("x", legend_start - 10)

      legend.select(".legend_mid_txt" + name)
        .attr("x", (main_graph_width/2)-10)




      // update svg output chart /////////////////////////////////////////////////////////

      svg_output.attr("width", width)
      x_output.range([ 0, output_graph_width]);
      y_output.range([height, 0]);

      xAxis_out.scale(x_output);
      yAxis_out.scale(y_output);

      svg_output_Xaxis.call(xAxis_out)
      svg_output_Yaxis.call(yAxis_out)

      svg_output.select(".x.axis_out" + name).call(xAxis_out).select(".domain").remove();
      // svg_output.select(".y.axis_out").call(yAxis_out)

      svg_output.select('.line_output' + name).attr("d", output_line(data_output))


      var adj_width = width - buffer
      // svg_output.attr("width", width)
      svg_output
        // .attr('width', width + 100)+
        .attr("transform",
          "translate(" + adj_width + "," + 0 + ")");

      svg_output.select(".output_svg_back" + name).attr('width', width)


      // special width resize for mobile screens ///////////////////////////////////////////
      // if (width <= 700) {
      //   output_graph_width = 
      // }


      }




// helper function to make dynamic legend that varies with screen size
function legend_size(current_width) {
  console.log('called')
  console.log(current_width)

  if (700 <= current_width && current_width <= 800) {
    console.log('1')
    legend_width = current_width - 100 
    legend_start = 100
  }
  else if (current_width < 700) {
    console.log('2')
    legend_width = current_width - 50
    legend_start = 50
  }
  else {
    console.log('3')
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
    main_graph_width = width - 112.5

  }
  else {
    output_graph_width = current_height / 2.75
    buffer = 120
    main_graph_width = width - 160    
  }


  return [main_graph_width, buffer, output_graph_width]
}
















