window.onload = function() {

    var chart = new CanvasJS.Chart("chartContainer", {
        animationEnabled: true,
        title: {
            text: ""
        },
        data: [{
        {% if dislike_chart == "bar" %}
            type: "bar",
        {% elif dislike_chart == "pie" %}
            type: "pie",
        {% elif dislike_chart == "spline" %}
            type: "spline",
        {% endif %}
            startAngle: 240,
            yValueFormatString: "##0.00\"%\"",
            indexLabel: "{label} {y}",
            dataPoints: [
            {% for o in form %}
                {y: {{o.dcount}}, label: "{{o.names}}"},
            {% endfor %}
            ]
        }]
    });
    chart.render();
    
    }
    