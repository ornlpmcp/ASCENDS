function set_selectable() {
  $('#input-cols-list li').click(function (e) {
      e.preventDefault()

      $that = $(this);

      if ($(this).hasClass('active')) {
        $(this).removeClass('active');
      }
      else {
        $that.addClass('active');
      }
    });
}

function load_data() {

  var data =
    {
      path_to_data: path_to_data
    };

  var dataToSend = JSON.stringify(data);

  $("#input-cols-list li").each(function (index) {
    $(this).remove();
  });
  
  $.ajax({
    url: '/open_file',
    type: 'POST',
    data: dataToSend,
    processData: false,
    contentType: false,

    success: function (jsonResponse) {
      var objresponse = JSON.parse(jsonResponse);
      table_loaded = objresponse;

      path_to_data = objresponse['path_to_data'];
      if (objresponse['msg'] == 'success') {

        var if_number = objresponse['if_number'];

        attr_content = '<ul class="list-group" id="table-attrs-list">'
        for (var key in if_number) {
          if (if_number[key] == true) {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-primary">Number</span></p></li>
                    `;
            cols_type[key] = 'Number';
          }
          else {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-danger">String</span></p></li>
                    `;
            cols_type[key] = 'String';
          }

        }
        attr_content += '</ul>'
        $('#table-attrs').html(attr_content);

        $(function () {
          console.log('ready');

          $('#table-attrs-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);

            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });
        });

      }

      else {
        $('#modal-title').html('Error');
        $('#modal-content').html('<div class="alert alert-danger" role="alert"> Something went wrong. Error code=' + objresponse['msg'] + '</div>');
        $('#my-modal').modal('show');

        $('#my-modal').on('hidden.bs.modal', function (e) {
          window.location = "/";
        });

      }

    },
    error: function (jsonResponse) {
      alert('Something went wrong.');
    }


  });

  if (input_cols.length > 0) {
    for (i in input_cols) {
      attr_content = `
                    <li class="list-group-item"><p class="text-right"><b>`+ input_cols[i] + `</b> <span class="label label-primary">Number</span></p></li>
                  `;
      $('#input-cols-list').append(attr_content);
      $('#input-cols').html('');
    }
  }


 set_selectable();
 
  if (target_col != "null" && target_col!=null) {
    target_col_html = `
              <p class="text-center"><b>`+ target_col + `</b> <span class="label label-primary">Number</span></p>        `
    $('#target-col').html(target_col_html);
  }
  else {
    target_col = null;
  }
};

function load_data_to_table() {

  var data =
    {
      path_to_data: path_to_data
    };

  var dataToSend = JSON.stringify(data);

  $("#input-cols-list li").each(function (index) {
    $(this).remove();
  });

  $.ajax({
    url: '/open_file',
    type: 'POST',
    data: dataToSend,
    processData: false,
    contentType: false,

    success: function (jsonResponse) {
      var objresponse = JSON.parse(jsonResponse);
      table_loaded = objresponse;

      path_to_data = objresponse['path_to_data'];
      if (objresponse['msg'] == 'success') {

        header_html = ''
        for (i = 0; i < objresponse['header'].length; i++) {
          header_html += '<th>' + objresponse['header'][i] + '</th>'
        }

        table_html = `
                  <table id="csv-table" class="display" cellspacing="0" width="100%" height="100%">
                  <thead>
                    <tr>
                      `+ header_html + `
                    </tr>
                  </thead>
                </table>
                `

        $('#table-content').html(table_html);

        try {
          $('#csv-table').dataTable().fnDestroy();
        }
        catch (err) {
          alert(err);
        }

        $('#csv-table').DataTable({
          "scrollX": true,
          "scrollY": "200px",
          "scrollCollapse": true,
          "paging": true
        });

        $('#csv-table').dataTable().fnClearTable();

        for (i = 0; i < objresponse['rows'].length; i++) {
          row = objresponse['rows'][i];
          $('#csv-table').dataTable().fnAddData(row);
        }

        var if_number = objresponse['if_number'];

        attr_content = '<ul class="list-group" id="table-attrs-list">'
        for (var key in if_number) {
          if (if_number[key] == true) {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-primary">Number</span></p></li>
                    `;
            cols_type[key] = 'Number';
          }
          else {
            attr_content += `
                      <li class="list-group-item"><p class="text-right"><b>`+ key + `</b> <span class="label label-danger">String</span></p></li>
                    `;
            cols_type[key] = 'String';
          }

        }
        attr_content += '</ul>'
        $('#table-attrs').html(attr_content);

        $(function () {
          console.log('ready');

          $('#table-attrs-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);

            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });
        });

        if (input_cols.length > 0) {
          for (i in input_cols) {
            attr_content = `
                    <li class="list-group-item"><p class="text-right"><b>`+ input_cols[i] + `</b> <span class="label label-primary">Number</span></p></li>
                  `;
            $('#input-cols-list').append(attr_content);
            $('#input-cols').html('');
          }
        }

         $('#input-cols-list li').click(function (e) {
            e.preventDefault()

            $that = $(this);

            if ($(this).hasClass('active')) {
              $(this).removeClass('active');
            }
            else {
              $that.addClass('active');
            }
          });

        if (target_col != "null" && target_col!=null) {
          target_col_html = `
              <p class="text-center"><b>`+ target_col + `</b> <span class="label label-primary">Number</span></p>
            `
          $('#target-col').html(target_col_html);
        }
        else {
          target_col = null;
        }

      }

      else {
        $('#modal-title').html('Error');
        $('#modal-content').html('<div class="alert alert-danger" role="alert"> Something went wrong. Error code=' + objresponse['msg'] + '</div>');
        $('#my-modal').modal('show');

        $('#my-modal').on('hidden.bs.modal', function (e) {
          window.location = "/";
        });

      }

    },
    error: function (jsonResponse) {
      alert('Something went wrong.');
    }


  });

};


function clear_table() {

  $('#table-content').html(`
      <div class="alert alert-info" role="alert">
        Please open a CSV file.
      </div>
      `);

  $('#table-attrs').html(`<div class="alert alert-info" role="alert">
        No contents to display.
      </div>`);

  $('#input-cols').html(`<div class="alert alert-info" role="alert">
        No input column has been selected.
      </div>`);

  $('#target-col').html(`<div class="alert alert-info" role="alert">
        No target column has been selected.
      </div>`);

  $('#corr-chart').html(``);

  $('#avail-chart-list').html(`<select class="form-control">
              <option>Chart not available.</option>
            </select>`);

  table_loaded = false;
  target_col = null;
  path_to_data = '';
  cols_type = {};
  input_cols = [];

}


String.prototype.unescapeHtml = function () {
  var temp = document.createElement("div");
  temp.innerHTML = this;
  var result = temp.childNodes[0].nodeValue;
  temp.removeChild(temp.firstChild);
  return result;
}

function add_to_target() {
  var col_to_add = [];
  var col_to_add_type = [];

  $("#table-attrs-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      var input_col_key = $(this).text().slice(0, -6).trim();
      var input_col_type = $(this).text().substring(input_col_key.length, $(this).text().length).trim();
      col_to_add.push(input_col_key);
      col_to_add_type.push(input_col_type);
    }
  });

  if (col_to_add.length == 1) {
    if (col_to_add_type[0] == 'Number') {
      target_col = col_to_add[0];
      cols_type[col_to_add[0]] = col_to_add_type[0];
      target_col_html = `
              <p class="text-center"><b>`+ col_to_add[0] + `</b> <span class="label label-primary">Number</span></p>
            `
      $('#target-col').html(target_col_html);
    } else {
      $('#modal-title').html('Warning');
      $('#modal-content').html('<div class="alert alert-danger" role="alert"> ' + 'Please select just one Number type column.' + '</div>');
      $('#my-modal').modal('show');
    }
  }
  else {
    $('#modal-title').html('Warning');
    $('#modal-content').html('<div class="alert alert-danger" role="alert"> ' + 'Please select just one Number type column.' + '</div>');
    $('#my-modal').modal('show');
  }
};

function add_to_input() {
  var not_added_list = []

  $("#table-attrs-list li").each(function (index) {

    if ($(this).hasClass('active')) {

      var input_col_key = $(this).text().slice(0, -6).trim();
      var input_col_type = $(this).text().substring(input_col_key.length, $(this).text().length).trim();

      if ($.inArray(input_col_key, input_cols) == -1) {
        if (input_col_type == 'String') {
          not_added_list.push(input_col_key);
        }
        else {
          input_cols.push(input_col_key);
          attr_content = `
                    <li class="list-group-item"><p class="text-right"><b>`+ input_col_key + `</b> <span class="label label-primary">Number</span></p></li>
                  `;
          $('#input-cols-list').append(attr_content);
          $('#input-cols').html('');
        }
      };

      $(this).removeClass('active');
    }
  });

  if (not_added_list.length > 0) {
    $('#modal-title').html('Warning');
    $('#modal-content').html('<div class="alert alert-warning" role="alert"> ' + 'String type columns:' + not_added_list + ' cannot be added to the input column list.' + '</div>');
    $('#my-modal').modal('show');
  }

  //attr_content+= ''


  if (input_cols.length == 0) {
    $('#input-cols').html(`<div class="alert alert-info" role="alert">
            No input column has been selected.
          </div>`);
  }
  $(function () {
    console.log('ready');

    $('#input-cols-list li').click(function (e) {
      e.preventDefault()

      $that = $(this);

      if ($(this).hasClass('active')) {
        $(this).removeClass('active');
      }
      else {
        $that.addClass('active');
      }
    });
  });

}

function select_all_input_cols() {
  $("#input-cols-list li").each(function (index) {
    $(this).addClass('active');
  });
}

function unselect_all_input_cols() {
  $("#input-cols-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      $(this).removeClass('active');
    }
  });
}

function unselect_all() {
  $("#table-attrs-list li").each(function (index) {
    if ($(this).hasClass('active')) {
      $(this).removeClass('active');
    }
  });
}

function select_all() {
  $("#table-attrs-list li").each(function (index) {
    $(this).addClass('active');
  });
}

function remove_input_cols() {
  $("#input-cols-list li").each(function (index) {

    if ($(this).hasClass('active')) {
      var input_col_key = $(this).text().slice(0, -6).trim();
      input_cols.splice(input_cols.indexOf(input_col_key), 1);
      $(this).removeClass('active');
      $(this).remove();

      $("#pred-input-cols div").each(function (index) {
        if(this.id=="form-"+input_col_key){
          $(this).remove();
        }
      });
    }
  });

  

  if (input_cols.length == 0) {
    $('#input-cols').html(`<div class="alert alert-info" role="alert">
            No input column has been selected.
          </div>`);
  }
}