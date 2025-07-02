import os
import neptune
import logging
import json
import constants

logger = logging.getLogger()
logger.setLevel(logging.INFO)

host = constants.neptune_endpoint
port = constants.neptune_port


def run_query(query):
    try:
        neptune_client = neptune.NeptuneClient(host, port)
        neptune_client.connect()
        status, response = neptune_client.execute_query(query)
        return status, response
    except Exception as e:
        raise e


def round_results(response):
    for record in response['results']:
        for key, value in record.items():
            if isinstance(value, (int, float)):
                record[key] = round(value, 2)

    return response


def is_table(response):
    result = None
    # print(f"result:- {result}")
    for result in response['results']:
        for values in result.values():
            if isinstance(values, dict) or isinstance(values, list):
                result = False
            else:
                result = True

    return result


def detect_query_data(response):
    result_set = set()
    for result in response['results']:
        for values in result.values():
            if isinstance(values, dict):
                if '~entityType' in values:
                    result_set.add(values['~entityType'])
                else:
                    result_set.add('table')
            elif isinstance(values, list):
                for value in values:
                    if '~entityType' in value:
                        result_set.add(value['~entityType'])
                    else:
                        result_set.add('table')
            else:
                result_set.add('path')

    return result_set

def node_object(data):
    node = {'id': data['~id'], 'label': (',').join(data['~labels']),
            '$$Name$$': data['~properties']['$$Name$$'],
            'properties': data['~properties']}

    return node


def edge_object(data):
    edge = {'id': data['~id'], 'label': data['~type'], 'properties': data['~properties'],
            'source': data["~start"], 'target': data["~end"]}

    return edge

def get_list(data, entity_type):
    vertex_r_edge_list = []
    table_data = None
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict):
                if values.get('~entityType'):
                    if entity_type == 'node':
                        vertex = node_object(values)
                        if vertex not in vertex_r_edge_list:
                            vertex_r_edge_list.append(vertex)
                    elif entity_type == 'relationship':
                        edge = edge_object(values)
                        vertex_r_edge_list.append(edge)
                elif entity_type == 'table':
                    table_data = data
            elif isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        if entity_type == value['~entityType'] == 'node':
                            vertex = node_object(value)
                            if vertex not in vertex_r_edge_list:
                                vertex_r_edge_list.append(vertex)
                        elif entity_type == value['~entityType'] == 'relationship':
                            edge = edge_object(value)
                            vertex_r_edge_list.append(edge)
                        elif entity_type == 'table':
                            table_data = data
            elif entity_type == 'path':
                return data['results'], table_data

    return vertex_r_edge_list, table_data


def create_vertex_for_edge(data):
    vertex_list = []
    for result in data['results']:
        for values in result.values():
            if isinstance(values, dict):
                vertex_list.append({'id': values['~start'], 'label': None, 'properties': None})
                vertex_list.append({'id': values['~end'], 'label': None, 'properties': None})
            else:
                logger.debug(f"Type of values: {type(values)}")

    return vertex_list


def process_response(response):
    try:
        if not response['results']:
            return {'table': response['results'], 'type': 'table'}

        node_r_edge = {
            'node': 'nodes',
            'relationship': 'edges',
            'path': 'path',
            'table': 'table'
        }
        table_data = None
        # response = round_results(response)

        if is_table(response):
            return {'table': response['results'], 'type': 'table'}

        results = detect_query_data(response)

        if len(results) == 1 and results.__contains__('relationship'):
            edge, table_data = get_list(response, 'relationship')
            vertex_for_edge = create_vertex_for_edge(response)
            return {'graph': {'nodes': vertex_for_edge, 'edges': edge}, 'type': 'graph'}

        elif len(results) == 2 and results.__contains__('path') and results.__contains__('table'):
            return {'table': response['results'], 'type': 'table'}

        data = {}
        for result in results:
            edge_r_node_list, table_data = get_list(response, result)
            data[node_r_edge.get(result)] = edge_r_node_list

        if data.get('path'):
            return {'path': data['path'], 'graph': {
                'nodes': data.get('nodes', []), 'edges': data.get('edges', [])
            }, 'type': 'path'}

        if table_data:
            return {'table': table_data, 'type': 'table'}

        return {'graph': {'nodes': data['nodes'], 'edges': data.get('edges', [])},
                'type': 'graph'}
    except Exception as e:
        raise e
